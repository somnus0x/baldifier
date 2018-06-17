import cv2
import numpy as np
from utils import Utils

def seamless_clone_face_head (head_image, placed_face_image, jaw_LR_points, brow_RL_Points, headPoints):
    expand_jaw_LR_points = __expand_points(jaw_LR_points)
    face_roi = np.vstack((expand_jaw_LR_points,brow_RL_Points)).astype(int)

    clone_image = __seamsless_clone(placed_face_image, head_image, face_roi)
    return clone_image

def __expand_points(points):

    expand_points = np.copy(points)
    counter = 0
    for point in expand_points:
        if(counter > 0  and counter < 5):
            point[0] -= 10
        elif(counter >= 5 and counter < 13 ):
            point[1] += 15
        elif(counter >= 13 and counter < 16 ):
            point[0] += 10
        counter += 1

    return expand_points

def __seamsless_clone(src, dst, mask_points):

    for point in mask_points:
        point[0] = min(point[0], dst.shape[1])
        point[1] = min(point[1], dst.shape[0])
        point[0] = max(point[0], 0)
        point[1] = max(point[1], 0)

    mask = np.zeros(src.shape, dtype=np.uint8)
    roi_corners = np.array(mask_points, dtype=np.int32)
    channel_count = src.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
    center_point = Utils.center_of_roi_rect(mask_points)
    cloned = cv2.seamlessClone(src, dst, mask, center_point, cv2.NORMAL_CLONE)
    return cloned