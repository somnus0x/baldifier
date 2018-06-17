from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import math


def __get_angle(startPoint, endPoint):

    x = endPoint[0] - startPoint[0]
    y = endPoint[1] - startPoint[1]

    return np.arctan2(y, x)

def align_face_points(jaw_LR_points, brow_RL_Points):

    angle = 360 - np.degrees(
        __get_angle(jaw_LR_points[0],
        jaw_LR_points[len(jaw_LR_points) - 1])
    )

    (h, w) = (100, 100)
    (cX,cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    ones = np.ones(shape=(len(jaw_LR_points), 1))
    points_ones = np.hstack([jaw_LR_points, ones])
    jaw_LR_points = M.dot(points_ones.T).T

    ones = np.ones(shape=(len(brow_RL_Points), 1))
    points_ones = np.hstack([brow_RL_Points, ones]) 
    brow_RL_Points = M.dot(points_ones.T).T

    return jaw_LR_points, brow_RL_Points
