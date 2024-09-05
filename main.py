from Object_Detector.detector import Detector

def main():

    model_name = input("Enter the model you want to use:")
    classes = input("Enter names of classes you want to detect (comma separated):")
    threshold = float(input("Enter threshold for model confidence:"))

    d = Detector(model_name)
    d.video_processing(0 , classes, threshold)

if __name__ == "__main__":
    main()
