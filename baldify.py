from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import imghdr
from timeit import default_timer as timer

from baldify_err import *
from baldify_face_extractor import FaceExtractor
from baldify_face_aligner import align_face_points
from baldify_face_trimmer import trim_face_and_points
from baldify_head_merger import find_face_head_map_points, place_face_to_head, transform_face_point
from baldify_head_face_seamless import seamless_clone_face_head
from utils import Utils

class Baldify:

    # Hard coded part
    dlib_model_path = './dlib-model.dat'
    head_image = cv2.imread('./head.png')
    bald_LR_points = np.vstack((
        [65, 250], [66, 221], [75, 187], [88, 155], [111, 123], [136, 100],
        [165, 78], [203, 59], [248, 45], [293, 42], [339, 49], [376, 62],
        [413, 85], [439, 104], [462, 127], [483, 157], [496, 185], [504, 218], [506, 250]
    ))

    # [65, 250], [66, 221], [75, 187], [88, 155], [111, 123], [136, 100],
    # [165, 78], [203, 59], [248, 45], [293, 42], [339, 49], [376, 62],
    # [413, 85], [439, 104], [462, 127], [483, 157], [496, 185], [504, 218], [506, 250]

    landmark_extractor = None
    
    def __init__(self):
        self.landmark_extractor = FaceExtractor(self.dlib_model_path)

    def baldify(self, image_data):

        # Check contains file
        if image_data == None or len(image_data) <= 0:
            raise BaldifyException(err_file_invalid)

        # Check file extension valid
        image_extension = imghdr.what(None, image_data)
        if image_extension != 'jpeg' and image_extension != 'png' and image_extension != 'gif':
            raise BaldifyException(err_file_invalid)

        # Decode PNG to opencv
        image_bytes = np.asarray(bytearray(image_data), dtype=np.uint8)
        image = cv2.imdecode(np.array(image_bytes), cv2.IMREAD_COLOR)

        # Extract face landmark
        image = imutils.resize(image, width=500)
        jaw_LR_points, brow_RL_Points = self.landmark_extractor.extract_face(image)

        # Trim face image and points
        image = trim_face_and_points(image, jaw_LR_points, brow_RL_Points)

        # Align face points
        aligned_jaw_LR_points, aligned_brow_RL_points = align_face_points(jaw_LR_points, brow_RL_Points)

        # Find mapping point to head
        mapLeftPoint, mapRightPoint = find_face_head_map_points(aligned_jaw_LR_points, aligned_brow_RL_points)
        
        # Transform point to using mapping points
        ref = np.vstack((mapLeftPoint, mapRightPoint))
        aligned_jaw_LR_points, aligned_brow_RL_points = transform_face_point(ref, aligned_jaw_LR_points, aligned_brow_RL_points, self.bald_LR_points)

        # Wrap affine
        image = place_face_to_head(self.head_image, image, jaw_LR_points, aligned_jaw_LR_points, aligned_brow_RL_points)

        # Seamless clone
        image = seamless_clone_face_head(self.head_image, image, aligned_jaw_LR_points, aligned_brow_RL_points, self.bald_LR_points)

        # Mask alpha 
        full_roi = np.vstack((aligned_jaw_LR_points, np.flipud(self.bald_LR_points)))
        image = Utils.mask_image(image, full_roi)

        # Compress PNG
        _, buff = cv2.imencode('.png', image)
        return buff