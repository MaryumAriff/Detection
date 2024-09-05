from ultralytics import YOLO
from Utility.utils import *
import cv2


class Detector:

    def __init__(self, model_name, classes_to_detect, threshold):

        self.model = YOLO('ObjectDetector/Model/' + model_name)
        self.threshold = threshold
        names = self.model.names
        self.class_indexes = get_class_indexes(classes_to_detect, names)


    def image_processing(self, image):  # Detect objs in 1 image

        object_detections = self.model.predict(source=image, conf=self.threshold, classes=self.class_indexes)  # results
        return object_detections

    def video_processing(self, video_source):  # Process a video stream

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            exit()

        while True:

            ret, frame = cap.read()  # Read frames
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Calling detector for one image
            detected_objects = self.image_processing(frame)

            # Calling visualization function
            frame = draw_bounding_box(detected_objects, frame)

            cv2.imshow('YOLOv8 Detection', frame) #output

            # Breaking the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()





