from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import math

from baldify_face_extractor import FaceExtractor
from utils import Utils

def find_face_head_map_points(jaw_LR_points, brow_RL_Points):
    target_y = min(
        brow_RL_Points[0][1],
        brow_RL_Points[len(brow_RL_Points)-1][1],
        jaw_LR_points[0][1],
        jaw_LR_points[len(jaw_LR_points)-1][1]
    )
    return [[jaw_LR_points[0][0], target_y], [jaw_LR_points[len(jaw_LR_points)-1][0], target_y]]

def transform_face_point (face_ref_points, jaw_LR_points, brow_RL_Points, bald_LR_points):
    bald_ref_points = (bald_LR_points[0], bald_LR_points[len(bald_LR_points)-1])
    
    bald_ref_distance = bald_ref_points[len(bald_ref_points)-1][0] - bald_ref_points[0][0]
    face_ref_distance = face_ref_points[1][0] - face_ref_points[0][0]

    bald_ref_center = (bald_ref_points[1] + bald_ref_points[0]) / 2
    face_ref_center = (face_ref_points[1] + face_ref_points[0]) / 2
    
    scale_ratio = float(bald_ref_distance) / float(face_ref_distance)

    jaw_LR_points = ((jaw_LR_points - face_ref_center) * scale_ratio) + bald_ref_center
    brow_RL_Points = ((brow_RL_Points - face_ref_center) * scale_ratio) + bald_ref_center
    return jaw_LR_points, brow_RL_Points

def place_face_to_head (dest_head_image, src_face_image, src_jaw_LR_points, dest_jaw_LR_points, dest_brow_RL_points):
    matrix = cv2.getAffineTransform(
        np.float32([src_jaw_LR_points[0], src_jaw_LR_points[len(src_jaw_LR_points)//2], src_jaw_LR_points[len(src_jaw_LR_points)-1]]),
        np.float32([dest_jaw_LR_points[0], dest_jaw_LR_points[len(src_jaw_LR_points)//2], dest_jaw_LR_points[len(src_jaw_LR_points)-1]])
    )
    dest_head_image = np.copy(dest_head_image)
    affine_face_image = cv2.warpAffine(src_face_image, matrix, (dest_head_image.shape[1], dest_head_image.shape[0]))
    face_roi = np.vstack((dest_jaw_LR_points,dest_brow_RL_points))
    affine_masked_image = Utils.mask_image(affine_face_image, face_roi)

    merged_image = Utils.blend_transparent(dest_head_image, affine_masked_image)
    return merged_image