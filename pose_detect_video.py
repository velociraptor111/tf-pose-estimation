import pickle

import numpy as np
import cv2

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from scipy.spatial import distance
from tf_pose import common
import argparse


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
constant_vals_dict[8] =0.2737839816086956
constant_vals_dict[9] = 0.43228599890116287
constant_vals_dict[10] = 0.6107515406413992
constant_vals_dict[11] = 0.2805108925623188
constant_vals_dict[12] = 0.4341653366521739
constant_vals_dict[13] = 0.6192064038720931


def get_l2_vector(humans,remove_idx=[]):
    total_l2_vector_dict = {}

    for human_id,human in enumerate(humans):
        anchor_body_part_id = 1 # ID of the neck
        print(human.body_parts)
        if anchor_body_part_id in human.body_parts:   #Check if neck is in the detection of openpose
            anchor_body_part_vector = [human.body_parts[anchor_body_part_id].x,human.body_parts[anchor_body_part_id].y]

            if anchor_body_part_id not in human.body_parts.keys():
                assert 1 == 2 # Neck is not detected

            l2_vector_array = []
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    l2_vector_array.append('NAN')
                else:
                    body_part = human.body_parts[i]
                    l2_distance  = distance.euclidean(anchor_body_part_vector, [body_part.x,body_part.y])
                    # print([body_part.x,body_part.y])
                    l2_vector_array.append(l2_distance)

            l2_vector_array = np.array(l2_vector_array)
            l2_vector_array = np.delete(l2_vector_array,remove_idx)
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

def classify_pose(l2_vector_arr,svm_classifier):
    prediction = svm_classifier.predict(l2_vector_arr)
    return prediction

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_video",default=0,help="Set to 1 if you want to save the video in a .avi format",type=int)
    parser.add_argument("--path_to_video",default='./test_video/time_up_go_one_person.mp4',help="Specify the path to your video")
    args = parser.parse_args()


    '''
        Tensorflow Graph declaration here 
    '''
    w, h = model_wh("432x368")
    model = "mobilenet_thin"
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))

    resize_out_ratio = 4
    PATH_TO_VIDEO = args.path_to_video

    with open('sitting_standing_clf.pkl', 'rb') as fid:
        svm_classifier = pickle.load(fid)

    cap = cv2.VideoCapture(PATH_TO_VIDEO)

    _, image_ori = cap.read()
    image_ori_height = image_ori.shape[0]
    image_ori_width = image_ori.shape[1]


    if args.save_video == 1:
        out = cv2.VideoWriter('time_up_go_one_person_pose_manipulated.avi', -1, 20.0, (image_ori_width, image_ori_height))

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    while cap.isOpened():
        ret_val, image = cap.read()

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
        total_l2_vector_dict = get_l2_vector(humans, remove_idx=[14, 15, 16, 17])

        prediction_dictionary = {}
        for human_id in total_l2_vector_dict:

            l2_vector_arr = total_l2_vector_dict[human_id]
            post_process_l2_vector(l2_vector_arr)

            l2_vector_arr = np.expand_dims(l2_vector_arr,axis=0)

            prediction_scores = classify_pose(l2_vector_arr, svm_classifier)

            for prediction_score in prediction_scores:
                if prediction_score == 0:
                    prediction_dictionary[human_id] = "Sitting"

                elif prediction_score == 1:
                    prediction_dictionary[human_id] = "Standing"


        ### Drawing to the image ###
        image_h, image_w = image.shape[:2]

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)


        for human_id,human in enumerate(humans):
            # If the person has no neck, then we know that there must also be NO prediction.
            if human_id in prediction_dictionary:
                neck_part = human.body_parts[1]
                cv2.putText(image, prediction_dictionary[human_id], (max(0,int((neck_part.x * image_w) - 150)), max(0,int((neck_part.y * image_h) - 100))), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
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



