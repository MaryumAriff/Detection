from ObjectDetector.detector import Detector

def main():

    model_name = 'yolov8n.pt'
    classes_to_detect = ['person', 'cup']
    threshold = 0.55

    d = Detector(model_name, classes_to_detect, threshold)
    d.video_processing(0)

if __name__ == "__main__":
    main()
