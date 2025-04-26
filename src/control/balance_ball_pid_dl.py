# -*- coding: utf-8 -*-
import signal
import sys
import cv2
import os
import json
from simple_pid import PID
from SphereDetectionNN.src.hardware.adxl345 import ADXL345
from SphereDetectionNN.src.hardware.MCP4728 import MCP4728
from time import sleep
from datetime import datetime
from SphereDetectionNN.src.trackers.ball_tracker_dl import BallTrackerDL

# Initialize sensor
adxl345 = ADXL345()
print("ADXL345 on address 0x%x:" % adxl345.address)

# Default setpoints
x_setpoint = 315
y_setpoint = 300

# Override setpoints from command line args if provided
if len(sys.argv) > 2:
    try:
        x_setpoint = float(sys.argv[1])
        y_setpoint = float(sys.argv[2])
        print("Using provided setpoints: X = {}, Y = {}".format(x_setpoint, y_setpoint))
    except ValueError:
        print("Invalid setpoint values. Using default.")

# Initialize PID controllers (tuned for DL tracker)
pid_x = PID(0.4, 0.2, 0.1, setpoint=x_setpoint)
pid_y = PID(0.4, 0.2, 0.1, setpoint=y_setpoint)

# Initialize DAC
dac_x4 = MCP4728(address=0x60, debug=True)
dac_x4.set_ext_vcc(channel=0, vcc=5.1)
dac_x4.set_ext_vcc(channel=1, vcc=5.1)
dac_x4.ch0_gain = 1
dac_x4.ch0_pd = 0
dac_x4.ch0_vref = 0
dac_x4.ch1_gain = 1
dac_x4.ch1_pd = 0
dac_x4.ch1_vref = 0

# Prepare output directories
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "recording_" + timestamp
img_dir = os.path.join(output_dir, "img")
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

seqinfo_path = os.path.join(output_dir, "seqinfo.ini")
labels_path = os.path.join(output_dir, "labels.json")
labels_txt_path = os.path.join(output_dir, "labels.txt")  
label_data = []
frame_count = 0
x, y = 0.0, 0.0

# Open labels.txt for continuous writing
labels_txt_file = open(labels_txt_path, 'w')  

def cleanup_and_exit(signal_received, frame, tracker):
    print("\nStopping... Setting DACs to 2.5V.")
    dac_x4.ch0_vout = 2.5
    dac_x4.ch1_vout = 2.5
    dac_x4.multi_write(ch0=True, ch1=True)
    tracker.release_camera()
    cv2.destroyAllWindows()

    # Write sequence info
    config_lines = [
        "[Sequence]",
        "name=circle_tracking_{}".format(timestamp),
        "imDir={}".format(img_dir),
        "frameRate=25",
        "seqLength={}".format(frame_count),
        "imWidth=800",
        "imHeight=600",
        "imExt=.jpg",
        "cameraID=1"
    ]
    with open(seqinfo_path, 'w') as f:
        f.write("\n".join(config_lines))
    print("Sequence info written:", seqinfo_path)

    # Save labels in JSON
    with open(labels_path, 'w') as jf:
        json.dump(label_data, jf, indent=2)
    print("Labels saved:", labels_path)

    # Save labels in TXT
    labels_txt_file.close() 
    print("Text labels saved:", labels_txt_path)

    print("Done.")
    sys.exit(0)

def main_loop():
    global x, y, frame_count
    last_x, last_y = 0.0, 0.0  # Low-pass smoothing
    alpha = 0.3  # Smoothing factor
    tracker = BallTrackerDL(model_path="best_yolov5_ball_model.pt", camera_index=1)   
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup_and_exit(sig, frame, tracker))

    try:
        while True:
            sleep(0.01)
            ball_position = tracker.get_ball_position()
            img = tracker.get_last_frame()

            if ball_position:
                # Apply smoothing to ball position
                raw_x, raw_y = ball_position
                x = alpha * raw_x + (1 - alpha) * last_x
                y = alpha * raw_y + (1 - alpha) * last_y
                last_x, last_y = x, y
            else:
                x, y = last_x, last_y

            # PID control
            y_x = pid_x(x) / 315.0
            y_y = pid_y(y) / 300.0
            vx = 2.5 + 2.5 * y_x - 2.5 * y_y
            vy = 2.5 + 2.5 * y_x + 2.5 * y_y
            vx = max(0, min(5, vx))
            vy = max(0, min(5, vy))

            print("X = {:.1f}, Y = {:.1f} | VCh1 = {:.2f} V, VCh2 = {:.2f} V".format(x, y, vx, vy))

            # Update DAC output
            dac_x4.ch0_vout = vx
            dac_x4.ch1_vout = vy
            dac_x4.multi_write(ch0=True, ch1=True)

            RECORD = False  # <--- switch to True if you want to save frames

            if img is not None:
                # Draw bounding box if available
                bbox = tracker.get_ball_bbox()
                if bbox:
                    x1, y1, w, h = bbox
                    cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                    cv2.putText(img, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Always display live tracking window
                cv2.imshow("Live Tracking", img)

                if RECORD:
                    frame_id = "{:06d}".format(frame_count)
                    frame_name = frame_id + ".jpg"
                    frame_path = os.path.join(img_dir, frame_name)
                    cv2.imwrite(frame_path, img)

                    if bbox:
                        label_data.append({
                            "frame": frame_id,
                            "bbox": [x1, y1, w, h],
                            "vx": round(vx, 2),
                            "vy": round(vy, 2)
                        })
                        labels_txt_file.write("{}, {}, {}, {}, {}\n".format(frame_id, x1, y1, w, h))

                    frame_count += 1

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cleanup_and_exit(None, None, tracker)

    except Exception as e:
        print("Error: {}".format(e))
        cleanup_and_exit(None, None, tracker)

if __name__ == "__main__":
    main_loop()
