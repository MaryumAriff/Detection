# from torch.utils.tensorboard.summary import video
from ultralytics import YOLO
import cv2
import os
import pandas


class Detector:

    def __init__(self):  # Constructor to Load our model

        self.model = YOLO('yolov8n.pt')

    def detection_on_one_image(self, image):  # Detect objects in one image/frame

        results = self.model(image)
        return results

    def detection_on_images_folder(self, image_folder):  # Process a folder of images

        results = []
        for filename in os.listdir(image_folder):

            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Construct the full image path
                image_path = os.path.join(image_folder, filename)

                # Load the image
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Error: Could not load image {filename}.")
                    continue

                # Calling detector for one image
                results.append(self.detection_on_one_image(image))

        return results

    def detection_on_video(self, video_source):  # Process a video stream

        results = []

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
            results.append(self.detection_on_one_image(frame))

            cv2.imshow('Dash Cam Feed', frame)

            # Breaking the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Releasing the Video Capture
        cap.release()
        cv2.destroyAllWindows()

        return results


# ----------------------------------------------------

# Making object of Detector class and calling its functions

d = Detector()
d.detection_on_video(0)
d.detection_on_images_folder(os.path.join(os.path.dirname(__file__), '..', 'Images'))

#------------------------------------------------------


