from ultralytics import YOLO
from Utility_Functions.utils import *
import cv2


class Detector:

    def __init__(self, model_name):
        self.model = YOLO('Object_Detector/Model/' + model_name)

    def image_processing(self, image, classes_to_detect, threshold):  # Detect objs in 1 image

        names = self.model.names
        class_indexes = get_class_indexes(classes_to_detect, names) # Func to get index from class name

        object_detections = self.model.predict(source=image, conf=threshold, classes=class_indexes)  # results

        return object_detections

    def video_processing(self, video_source, classes, threshold):  # Process a video stream

        detected_objects = []
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print("Error: Could not open video source.")
            exit()

        while True:

            ret, frame = cap.read()  # Read frames

            if not ret:
                print("Error: Failed to capture frame.")
                break


            #Correcting format of classes
            classes = classes.lower()
            classes_to_detect = classes.split(',')

            #Correcting format of threshold
            threshold = round(threshold, 2)

            # Calling detector for one image
            detected_objects = self.image_processing(frame, classes_to_detect, threshold)

            # Calling visualization function
            frame=draw_bounding_box(detected_objects, frame)

            cv2.imshow('YOLOv8 Detection', frame) #output

            # Breaking the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()





