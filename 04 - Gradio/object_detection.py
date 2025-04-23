from fastrtc import Stream
import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv10 model (Pretrained on COCO dataset)
model = YOLO("yolov10n.pt")

def detect_objects(frame):
# Convert frame (BGR) to RGB (YOLO expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run YOLO object detection
    # Returing an object from class ultralytics.engine.results.Results

    results = model(frame_rgb)[0]    

    # Draw bounding boxes on the frame
    # Boxes in [x1, y1, x2, y2] format

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])  # Convert box coordinates into int
        # Using cv2.rectangle(image, start_point, end_point, color, thickness) to draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame  # Return frame with detections

# FastRTC Stream for real-time video processing
stream = Stream(
    handler=detect_objects,  # Object detection function
    modality="video",  # Video mode
    mode="send-receive",  # Sends video, receives processed video
)

# Launch Gradio UI
stream.ui.launch()


