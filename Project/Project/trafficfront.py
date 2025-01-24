def detect_vehicles(images):

    model = YOLO("../Test Yolo/Yolo_weights/yolov8m.pt")

    img = cv2.imread("/Users/starkz/PycharmProjects/Yolo_object_detection/Project/Traffic Images/side_2/side_2.jpg")
    imgreg = img
    res = model(imgreg, stream=True)

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    detections = np.empty((0, 5))
    for r in res:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            confi = round(float(box.conf[0] * 100))
            cls = box.cls[0]
            current_class = classNames[int(cls)]

            if current_class in ["car", "motorbike", "bus", "truck"] and confi >= 30:
                current_array = np.array([x1, y1, x2, y2, confi])
                detections = np.vstack((detections, current_array))

    vehicle_count = len(detections)
    print("Count =", vehicle_count)

    for det in detections:
        x1, y1, x2, y2, confi = det
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(img, "Vehicle", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.putText(img, f'Count: {vehicle_count}', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return vehicle_count