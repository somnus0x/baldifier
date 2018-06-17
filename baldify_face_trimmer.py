from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import sys
from utils import Utils

def trim_face_and_points (faceImage, jaw_LR_points, brow_RL_Points):
    roi = np.vstack((jaw_LR_points,brow_RL_Points))
    min_point, max_point = Utils.rect_from_roi(roi)

    max_point[0] = min(max_point[0], faceImage.shape[1])
    max_point[1] = min(max_point[1], faceImage.shape[0])
    min_point[0] = max(min_point[0], 0)
    min_point[1] = max(min_point[1], 0)

    for point in jaw_LR_points:
        point[0] -= min_point[0]
        point[1] -= min_point[1]

    for point in brow_RL_Points:
        point[0] -= min_point[0]
        point[1] -= min_point[1]

    return faceImage[min_point[1]:max_point[1],min_point[0]:max_point[0]]
