# -*- coding: utf-8 -*-
import os
import json
import cv2

# Adjust this to your folder
recording_dir = r"PathtoYourRecordingDirectory"  # e.g. "\SphereDetectionNN\data\recordings\recording_20250514_155947.zip"
img_dir = os.path.join(recording_dir, "img")
label_path = os.path.join(recording_dir, "labels.json")

# Load labels
with open(label_path, 'r') as f:
    labels = json.load(f)

print("Showing {} images with bounding boxes...".format(len(labels)))

for item in labels:
    frame = item["frame"]
    bbox = item["bbox"]
    img_path = os.path.join(img_dir, frame + ".jpg")

    img = cv2.imread(img_path)
    if img is None:
        print("Error: could not load image {}".format(img_path))
        continue

    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h

    # Draw bounding box + frame ID
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, frame, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
    cv2.imshow("Bounding Box Viewer", img)

    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
