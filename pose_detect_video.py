import pickle

import numpy as np
import cv2

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from scipy.spatial import distance
from tf_pose import common
import argparse

constant_vals_dict = {}
constant_vals_dict[2] = 0.026407283590116293
constant_vals_dict[3] = 0.1587968211930503
constant_vals_dict[4] = 0.26106975708085106
constant_vals_dict[6] = 0.17225851005610576
constant_vals_dict[7] = 0.2688328088482758
constant_vals_dict[9] = 0.43228599890116287
constant_vals_dict[10] = 0.6107515406413992
constant_vals_dict[13] = 0.6192064038720931


def get_l2_vector(human,remove_idx=[]):
    human = human[0] # assume only one human
    # print(human.body_parts.keys())
    anchor_body_part_id = 1 # ID of the neck
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

    print(l2_vector_array)

    l2_vector_array = np.array(l2_vector_array)
    l2_vector_array = np.delete(l2_vector_array,remove_idx)
    # for removal_id in remove_idx:
    #     print(removal_id)
        # l2_vector_array.pop(removal_id)


    return l2_vector_array

def post_process_l2_vector(l2_vector_arr):
    '''
    Fill in the 'NAN' string with the average of that specific value in the dataset.
    :param l2_vector_arr: List of l2 distances specifically in the ID
    :return:
    '''

    l2_vector_arr_numpy = np.array(l2_vector_arr)
    index_nan = np.where(l2_vector_arr_numpy == "NAN")
    print(index_nan)
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

        human = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
        l2_vector_arr = get_l2_vector(human, remove_idx=[14, 15, 16, 17])
        post_process_l2_vector(l2_vector_arr)
        l2_vector_arr = np.expand_dims(l2_vector_arr,axis=0)
        prediction_score = classify_pose(l2_vector_arr, svm_classifier)
        print(prediction_score)
        score_string = ""
        if prediction_score[0] == 0:
            print("Sitting")
            score_string="Sitting"
        elif prediction_score[0] == 1:
            print("Standing")
            score_string="Standing"

        ### Drawing to the image ###
        image_h, image_w = image.shape[:2]

        neck_part = human[0].body_parts[1] #neck location
        image = TfPoseEstimator.draw_humans(image, human, imgcopy=False)
        cv2.putText(image, score_string, (max(0,int((neck_part.x * image_w) - 150)), max(0,int((neck_part.y * image_h) - 100))), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 255), 2)

        if args.save_video == 1:
            out.write(image)
        else:
            cv2.imshow('tf-pose-estimation result', image)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()



