import pickle
import sys
import os
import time
import math

import numpy as np
import tensorflow as tf
import cv2
from scipy.spatial import distance

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from scipy.spatial import distance
from tf_pose import common
import argparse


facenet_path = os.path.join('/Users/petertanugraha/Projects/tf-deep-facial-recognition-lite','facenet','src')
tf_deep_facial_recognition_path = '/Users/petertanugraha/Projects/tf-deep-facial-recognition-lite'


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0,path)
    else:
        print("Path already exist, not inserting anymore")


### The ordering here is sketchy but I need to give highest priority to the facenet path ###
add_path(tf_deep_facial_recognition_path)
add_path(facenet_path)

from src.align_image_mtcnn import align_image_with_mtcnn_with_tf_graph
from src.utils import *

### Average distances calculated ###
constant_vals_dict = {}
constant_vals_dict[0] = 0.08606471351014493
constant_vals_dict[1] = 0.0
constant_vals_dict[2] = 0.026407283590116293
constant_vals_dict[3] = 0.1587968211930503
constant_vals_dict[4] = 0.26106975708085106
constant_vals_dict[5] = 0.021211341402898553
constant_vals_dict[6] = 0.17225851005610576
constant_vals_dict[7] = 0.2688328088482758
constant_vals_dict[8] = 0.2737839816086956
constant_vals_dict[9] = 0.43228599890116287
constant_vals_dict[10] = 0.6107515406413992
constant_vals_dict[11] = 0.2805108925623188
constant_vals_dict[12] = 0.4341653366521739
constant_vals_dict[13] = 0.6192064038720931


def get_l2_vector(humans, remove_idx=[]):
    total_l2_vector_dict = {}

    for human_id, human in enumerate(humans):
        anchor_body_part_id = 1  # ID of the neck
        print(human.body_parts)
        if anchor_body_part_id in human.body_parts:  # Check if neck is in the detection of openpose
            anchor_body_part_vector = [human.body_parts[anchor_body_part_id].x, human.body_parts[anchor_body_part_id].y]

            if anchor_body_part_id not in human.body_parts.keys():
                assert 1 == 2  # Neck is not detected

            l2_vector_array = []
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    l2_vector_array.append('NAN')
                else:
                    body_part = human.body_parts[i]
                    l2_distance = distance.euclidean(anchor_body_part_vector, [body_part.x, body_part.y])
                    # print([body_part.x,body_part.y])
                    l2_vector_array.append(l2_distance)

            l2_vector_array = np.array(l2_vector_array)
            l2_vector_array = np.delete(l2_vector_array, remove_idx)
            total_l2_vector_dict[human_id] = l2_vector_array

    return total_l2_vector_dict


def post_process_l2_vector(l2_vector_arr):
    '''
    Fill in the 'NAN' string with the average of that specific value in the dataset.
    :param l2_vector_arr: List of l2 distances specifically in the ID
    :return:
    '''

    l2_vector_arr_numpy = np.array(l2_vector_arr)
    index_nan = np.where(l2_vector_arr_numpy == "NAN")

    for idx in index_nan[0]:
        if idx not in constant_vals_dict.keys():
            print("Never seen this nan variable within column: ", idx)
            assert 10 == 0
        else:
            # Replace the value NAN with a constant variable found from before.
            l2_vector_arr[idx] = constant_vals_dict[idx]


def calculate_face_embedding_similarity(face_embeddings_ground_zero,face_embedding):
    dis_to_face_0 = distance.euclidean(face_embeddings_ground_zero[0],face_embedding)
    dis_to_face_1 = distance.euclidean(face_embeddings_ground_zero[1], face_embedding)

    if dis_to_face_0 < dis_to_face_1:
        return 0
    else:
        return 1



def classify_pose(l2_vector_arr, svm_classifier):
    prediction = svm_classifier.predict(l2_vector_arr)
    return prediction


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_video", default=0, help="Set to 1 if you want to save the video in a .avi format",
                        type=int)
    parser.add_argument("--path_to_video", default='./test_video/chair_stand_original_trimmed.mp4',
                        help="Specify the path to your video")
    args = parser.parse_args()

    CROP_SSD_PERCENTAGE = 0.3
    FACENET_PREDICTION_BATCH_SIZE = 90
    IMAGE_SIZE = 160

    face_embeddings_ground_zero = np.load('face_embeddings.npy')

    '''
    
        Computation Graph Declarations Here
    
    '''
    w, h = model_wh("432x368")
    model = "mobilenet_thin"
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))

    resize_out_ratio = 4

    ### Load SSD Model to Memory ###
    PATH_TO_SSD_CKPT = '/Users/petertanugraha/Projects/tf-deep-facial-recognition-lite/model/frozen_inference_graph_custom.pb'
    FACENET_MODEL_PATH = '/Users/petertanugraha/Projects/tf-deep-facial-recognition-lite/facenet/models/facenet/20180402-114759/20180402-114759.pb'

    image_tensor, boxes_tensor, scores_tensor, \
                    classes_tensor, num_detections_tensor = load_tf_ssd_detection_graph(PATH_TO_SSD_CKPT)

    PATH_TO_VIDEO = args.path_to_video

    sess = tf.Session()

    with sess.as_default():

        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

        images_placeholder, embeddings, phase_train_placeholder = load_tf_facenet_graph(FACENET_MODEL_PATH)


        with open('sitting_standing_clf.pkl', 'rb') as fid:
            svm_classifier = pickle.load(fid)

        cap = cv2.VideoCapture(PATH_TO_VIDEO)

        _, image_ori = cap.read()
        image_ori_height = image_ori.shape[0]
        image_ori_width = image_ori.shape[1]

        if args.save_video == 1:
            out = cv2.VideoWriter('chair_stand_test.avi', -1, 20.0,
                                  (image_ori_width, image_ori_height))

        if cap.isOpened() is False:
            print("Error opening video stream or file")

        while cap.isOpened():
            ret_val, image = cap.read()

            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image from BGR to RGB to pass on SSD detector
            image_np_expanded = np.expand_dims(image_np, axis=0)

            start_time_ssd_detection = time.time()
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes_tensor, scores_tensor, classes_tensor, num_detections_tensor],
                feed_dict={image_tensor: image_np_expanded})
            elapsed_time = time.time() - start_time_ssd_detection
            print('SSD inference time cost: {}'.format(elapsed_time))

            dets = post_process_ssd_predictions(boxes, scores, classes)

            im_height = image.shape[0]
            im_width = image.shape[1]

            bbox_dict = {}
            ids = []
            images_array = []

            for detection_id, cur_det in enumerate(dets):
                boxes = cur_det[:4]
                (ymin, xmin, ymax, xmax) = (boxes[0] * im_height, boxes[1] * im_width,
                                            boxes[2] * im_height, boxes[3] * im_width)
                bbox = (xmin, xmax, ymin, ymax)
                new_xmin, new_xmax, new_ymin, new_ymax = crop_ssd_prediction(xmin, xmax, ymin, ymax,
                                                                             CROP_SSD_PERCENTAGE, im_width,
                                                                             im_height)
                roi_cropped_rgb = image_np[new_ymin:new_ymax, new_xmin:new_xmax]
                faces_roi, _ = align_image_with_mtcnn_with_tf_graph(roi_cropped_rgb, pnet, rnet, onet,
                                                                    image_size=IMAGE_SIZE)

                if len(faces_roi) != 0:  # This is either a face or not a face
                    faces_roi = faces_roi[0]
                    images_array.append(prewhiten(faces_roi))
                    ids.append(detection_id)
                    bbox_dict[detection_id] = bbox

            nrof_images = len(bbox_dict)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / FACENET_PREDICTION_BATCH_SIZE))
            emb_array = get_face_embeddings(sess, embeddings, images_placeholder, phase_train_placeholder,
                                            nrof_images, nrof_batches_per_epoch, FACENET_PREDICTION_BATCH_SIZE,
                                            images_array)

            print("PRINTING EMBEDDING ARRAY!")
            print(emb_array.shape)

            prediction_class_face = {}
            for idx,embedding in enumerate(emb_array):
                prediction_class_face[ids[idx]] = calculate_face_embedding_similarity(face_embeddings_ground_zero,embedding)

            print(prediction_class_face)

            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
            total_l2_vector_dict = get_l2_vector(humans, remove_idx=[14, 15, 16, 17])

            prediction_dictionary = {}
            for human_id in total_l2_vector_dict:

                l2_vector_arr = total_l2_vector_dict[human_id]
                post_process_l2_vector(l2_vector_arr)

                l2_vector_arr = np.expand_dims(l2_vector_arr, axis=0)

                prediction_scores = classify_pose(l2_vector_arr, svm_classifier)

                for prediction_score in prediction_scores:
                    if prediction_score == 0:
                        prediction_dictionary[human_id] = "Sitting"

                    elif prediction_score == 1:
                        prediction_dictionary[human_id] = "Standing"

            ### Drawing to the image ###
            image_h, image_w = image.shape[:2]

            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)


            for i,id in enumerate(ids):
                bbox = bbox_dict[id]
                if prediction_class_face[id] == 1:
                    input_string = "Patient A"
                else:
                    input_string = "Nurse"

                cv2.putText(image,input_string,(int(bbox[0]) - 20, int(bbox[2]) - 30), 0, 0.8, (255, 0, 0), thickness=2)
                cv2.rectangle(image, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), (255, 0, 0), 2)

            for human_id, human in enumerate(humans):
                # If the person has no neck, then we know that there must also be NO prediction.
                if human_id in prediction_dictionary:
                    neck_part = human.body_parts[1]
                    cv2.putText(image, prediction_dictionary[human_id], (
                    max(0, int((neck_part.x * image_w) - 150)), max(0, int((neck_part.y * image_h) - 100))),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                                (0, 0, 255), 2)

            if args.save_video == 1:
                out.write(image)
            else:
                cv2.imshow('tf-pose-estimation result', image)
                if cv2.waitKey(1) == 27:
                    break

        print("DONE Processing the video")
        cap.release()
        cv2.destroyAllWindows()



