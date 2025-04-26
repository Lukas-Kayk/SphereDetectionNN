# -*- coding: utf-8 -*-
import os
import cv2

# --- Adjust paths ---
recording_dir = r"PathtoYourRecordingDirectory"  # e.g. "\SphereDetectionNN\data\recordings\recording_20250514_155947.zip"
img_dir = os.path.join(recording_dir, "img")
txt_path = os.path.join(recording_dir, "labels.txt")

# Export folder (created automatically)
export_dir = os.path.join(recording_dir, "exports")
os.makedirs(export_dir, exist_ok=True)

# --- Load labels ---
with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

print("Showing {} images with bounding boxes...".format(len(lines)))
print("Keys: [s]=Save  [q]=Quit  [any other key]=Next")

def find_image_path(img_dir, frame_id):
    """Find image file for frame_id in img_dir (jpg/png)."""
    frame_id = frame_id.strip()
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(img_dir, frame_id + ext)
        if os.path.exists(p):
            return p
    return None

for line in lines:
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) != 5:
        # optionally try semicolon delimiter
        parts = [p.strip() for p in line.strip().split(";")]
        if len(parts) != 5:
            continue

    frame_id, x1, y1, w, h = parts
    try:
        x1 = int(x1); y1 = int(y1); w = int(w); h = int(h)
    except ValueError:
        continue

    x2, y2 = x1 + w, y1 + h

    img_path = find_image_path(img_dir, frame_id)
    if img_path is None:
        print("Image not found (jpg/png):", os.path.join(img_dir, frame_id + ".*"))
        continue

    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image:", img_path)
        continue

    # Draw bounding box + overlay text
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{frame_id}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "s=save, q=quit", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow("Label Viewer", img)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save with same basename + _boxed.jpg in exports folder
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(export_dir, f"{base}_boxed.jpg")
        ok = cv2.imwrite(out_path, img)
        if ok:
            print(f"Saved: {out_path}")
        else:
            print(f"Failed to save: {out_path}")
    # any other key: just continue

cv2.destroyAllWindows()
