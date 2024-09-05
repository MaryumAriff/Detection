import cv2

def draw_bounding_box(result, frame):
    boxes = result[0].boxes

    for box in boxes:
        # Extract coordinates and class
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]  # Confidence Score (Out of 1)
        cls = box.cls[0]   # Class Index (Label)

        # Draw the bounding box and label
        title = f'{result[0].names[int(cls)]} {conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, title, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
    # cv2.imshow('YOLOv8 Detection', frame)

    return frame

def get_class_indexes(classes_to_detect, names):

    class_indexes = []

    for c in classes_to_detect:
        for key, value in names.items():
            if value == c:
                class_indexes.append(key)

    return class_indexes