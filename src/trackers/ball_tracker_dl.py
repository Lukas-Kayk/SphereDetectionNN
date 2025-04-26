import torch
import cv2
import numpy as np

class BallTrackerDL:
    def __init__(self, model_path, camera_index=1, frame_width=800, frame_height=600):
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        # Load YOLOv5 model (custom weights)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

        self.last_img = None
        self.last_detection = None

    def get_ball_position(self):
        # Capture frame
        success, img = self.cap.read()
        if not success:
            return None

        # Flip camera orientation
        img = cv2.rotate(img, cv2.ROTATE_180)
        self.last_img = img.copy()

        # Run inference
        results = self.model(img)
        predictions = results.xyxy[0].cpu().numpy()

        if len(predictions) == 0:
            self.last_detection = None
            return None

        # Select the largest bounding box (if multiple detected)
        best = max(predictions, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        x1, y1, x2, y2, conf, cls = best
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Store last bounding box as (x, y, width, height)
        self.last_detection = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        return (cx, cy)

    def get_last_frame(self):
        return self.last_img

    def get_ball_bbox(self):
        return self.last_detection

    def release_camera(self):
        self.cap.release()
