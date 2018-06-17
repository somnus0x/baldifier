from imutils import face_utils
from baldify_err import *
import numpy as np
import argparse
import imutils
import dlib
import cv2

class FaceExtractor:

    __detector = None
    __predictor = None

    def __init__(self, dlib_model_path):

        self.__detector = dlib.get_frontal_face_detector()
        self.__predictor = dlib.shape_predictor(dlib_model_path)

    def extract_face(self, face_image):
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        faceRegions, scores, _ = self.__detector.run(gray, 2)

        if len(scores) < 1:
            raise BaldifyException(err_face_invalid)

        maxIndex = self.__findMaxIndex(scores)
        faceRegion = faceRegions[maxIndex]
        faceRoi = self.__predictor(gray, faceRegion)

        faceRoi = face_utils.shape_to_np(faceRoi)
        (jaw_start_index, jaw_end_index) = face_utils.FACIAL_LANDMARKS_IDXS['jaw']
        (r_brow_start_index, r_brow_end_index) = face_utils.FACIAL_LANDMARKS_IDXS['right_eyebrow']
        (l_brow_start_index, l_brow_end_index) = face_utils.FACIAL_LANDMARKS_IDXS['left_eyebrow']

        l_brow = faceRoi[l_brow_start_index:l_brow_end_index]
        l_brow = np.flip(l_brow, 0)

        r_brow = faceRoi[r_brow_start_index:r_brow_end_index]
        r_brow = np.flip(r_brow, 0)

        jaw_LR_points = faceRoi[jaw_start_index:jaw_end_index]
        brow_RL_Points = np.vstack((l_brow, r_brow))
        return jaw_LR_points, brow_RL_Points

    def __findMaxIndex(self, numbers):
        max = 0
        maxIndex = -1
        for (i, number) in enumerate(numbers):
            if (numbers[i] > max):
                max = number
                maxIndex = i
        return maxIndex