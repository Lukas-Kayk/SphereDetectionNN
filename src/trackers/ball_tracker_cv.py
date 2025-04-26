# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np

class BallTracker:
    def __init__(self, camera_index=1, frame_width=800, frame_height=600):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        self.crop = 50
        self.bbox = 120
        self.ball = None
        self.last_detection_time = 0
        self.last_roi = None
        self.roi_reset_time = 0.5
        self.last_img = None
        self.last_ellipse_size = None  

    def get_ball_position(self):
        success, img = self.cap.read()
        if not success:
            return None

        # Flip camera orientation
        img = cv2.rotate(img, cv2.ROTATE_180)
        self.last_img = img.copy()  # keep copy for later display/saving

        height, width = img.shape[:2]

        # Mask corners (remove irrelevant regions)
        img[:90, :180] = [128, 128, 128]
        img[height-95:, :170] = [128, 128, 128]
        img[:100, width-150:] = [128, 128, 128]
        img[height-95:, width-165:] = [128, 128, 128]

        # Crop borders
        crop_top, crop_left = 0, 86
        crop_bottom, crop_right = 0, 70
        img = img[crop_top:img.shape[0] - crop_bottom, crop_left:img.shape[1] - crop_right]
        self.last_img = img.copy()

        time_since_detection = time.time() - self.last_detection_time

        # Define ROI (region of interest)
        if self.ball is not None:
            roi = [
                (max(self.ball[0] - self.bbox, 0), max(self.ball[1] - self.bbox, 0)),
                (self.ball[0] + self.bbox, self.ball[1] + self.bbox)
            ]
        else:
            if time_since_detection > self.roi_reset_time:
                roi = ((self.crop, self.crop), (img.shape[1] - self.crop, img.shape[0] - self.crop))
            else:
                roi = self.last_roi

        roi_img = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

        # Preprocess ROI
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        edges = cv2.Canny(gray, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_ball = None

        # Find largest contour and fit ellipse
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if len(largest_contour) >= 5 and area > 150:
                ellipse = cv2.fitEllipse(largest_contour)
                center_x, center_y = ellipse[0]
                self.last_ellipse_size = ellipse[1]  # store size
                detected_ball = (int(center_x + roi[0][0]), int(center_y + roi[0][1]))
                self.last_roi = roi
                self.last_detection_time = time.time()

        if detected_ball is not None:
            self.ball = detected_ball
            return detected_ball
        else:
            self.ball = None
            self.last_ellipse_size = None  # size no longer valid
            return None

    def get_last_frame(self):
        return self.last_img

    def release_camera(self):
        self.cap.release()

    def get_ball_bbox(self):
        if self.ball is None or self.last_ellipse_size is None:
            return None
        x, y = self.ball
        w, h = self.last_ellipse_size
        w = max(w, 20)
        h = max(h, 20)
        x1 = int(max(0, x - w / 2))
        y1 = int(max(0, y - h / 2))
        return (x1, y1, int(w), int(h))
