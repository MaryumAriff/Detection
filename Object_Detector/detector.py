# from torch.utils.tensorboard.summary import video
from ultralytics import YOLO
from Object_Visualizer import visualize
import cv2
import os
import pandas


class Detector:

    def __init__(self):  # Constructor to Load our model
        self.model = YOLO('Object_Detector/Model/yolov8n.pt')

    def image_detection(self, image):  # Detect objects in one image/frame
        object_detections = self.model(image)
        return object_detections

    def video_processing(self, video_source):  # Process a video stream

        cap = cv2.VideoCapture(video_source)  # Starting Video Capture

        if not cap.isOpened():
            print("Error: Could not open video source.")
            exit()

        while True:

            ret, frame = cap.read()  # Read frames

            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Calling detector for one image
            result = self.image_detection(frame)

            # Calling visualizing function
            visualize.draw_box(result,frame)

            # Breaking the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Releasing the Video Capture
        cap.release()
        cv2.destroyAllWindows()






