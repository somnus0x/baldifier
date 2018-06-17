from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import sys

class Utils:

    @staticmethod
    def visualize(image, points):
        counter = 0
        for (x, y) in points:
            if(x > image.shape[1] or y > image.shape[0]):
                continue
            cv2.circle(image, (x, y), 3, (57, 255, 20), -1)

        cv2.imshow("visualize", image)
        cv2.waitKey(0)

    @staticmethod
    def center_of_roi_rect(roi):
        min, max = Utils.rect_from_roi(roi)
        return ((min[0] + max[0]) // 2, (min[1] + max[1]) // 2)

    @staticmethod
    def rect_from_roi (roi):
        min_point = [sys.maxsize, sys.maxsize]
        max_point = [0, 0]
        for point in roi:
            if min_point[0] > point[0]:
                min_point[0] = point[0]
            if min_point[1] > point[1]:
                min_point[1] = point[1]
            if max_point[0] < point[0]:
                max_point[0] = point[0]
            if max_point[1] < point[1]:
                max_point[1] = point[1]
        return min_point, max_point

    @staticmethod
    def mask_image (image, roi):

        # If channels are 3
        if(image.shape[2] == 3):
            image = Utils.__add_alpha_channel(image)
        
        # mask defaulting to black for 3-channel and transparent for 4-channel
        # (of course replace corners with yours
        mask = np.zeros(image.shape, dtype=np.uint8)
        roi_corners = np.array(roi, dtype=np.int32)
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image

        ignore_mask_color = (255,)*channel_count
        cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
        
        # apply the mask
        masked_image = cv2.bitwise_and(image, mask)

        return masked_image

    @staticmethod
    def __add_alpha_channel(image):

        tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(image)
        rgba = [b, g, r, alpha]
        dst = cv2.merge(rgba, 4)

        return dst

    @staticmethod
    def blend_transparent(base_image, overlay_image):

        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGRA2BGR)

        # Split out the transparency mask from the colour info
        overlay_img = overlay_image[:,:,:3] # Grab the BRG planes
        overlay_mask = overlay_image[:,:,3:]  # And the alpha plane

        # Again calculate the inverse mask
        background_mask = 255 - overlay_mask

        # Turn the masks into three channel, so we can use them as weights
        overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
        background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

        # Create a masked out face image, and masked out overlay
        # We convert the images to floating point in range 0.0 - 1.0
        face_part = (base_image * (1 / 255.0)) * (background_mask * (1 / 255.0))
        overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

        # And finally just add them together, and rescale it back to an 8bit integer image
        return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

    # @staticmethod
    # def fill_convex_poly_from_polygon(image, points, fill_color):
    #     subdiv = cv2.Subdiv2D()
    #     for point in points:
    #         print(point)
    #         subdiv.insert(point)
    #     # triangleList = subdiv.getTriangleList()
    #     # for t in triangleList :
    #         # print(t)
    #         # poly = np.vstack(((t[0], t[1]), (t[2], t[3]), (t[4], t[5])))
            # cv2.fillConvexPoly(image, poly, fill_color)