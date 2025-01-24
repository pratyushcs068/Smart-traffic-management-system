from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

chigari_model = YOLO("/Users/starkz/PycharmProjects/Yolo_object_detection/Test Yolo/Yolo_weights/Chigari.pt")
vehicle_model = YOLO("/Users/starkz/PycharmProjects/Yolo_object_detection/Test Yolo/Yolo_weights/yolov10s.pt")
emergency_model = YOLO("/Users/starkz/PycharmProjects/Yolo_object_detection/customtrain/runs/detect/train5/weights/best.pt")

vehicle_classes = ["car", "motorbike", "bus", "truck"]
emergency_classes = ["ambulance", "firetruck"]


def detect_objects(model, image_path, valid_classes, confidence_threshold=0.3):

    results = model(image_path)
    detections = results[0].boxes
    count = 0

    for box in detections:
        cls_id = int(box.cls[0])
        confidence = box.conf[0]
        class_name = model.names[cls_id]

        if class_name in valid_classes and confidence >= confidence_threshold:
            count += 1

    return count


def process_folder(folder_path, model, valid_classes):

    counts = []
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if os.path.isfile(image_path):
            count = detect_objects(model, image_path, valid_classes)
            counts.append(count)

    return max(counts, default=0)


def calculate_green_time(vehicle_count):

    if vehicle_count == 0:
        return 10
    return min(7 * 2 + (vehicle_count - 2) * 2, 120)


def traffic_signal_logic(folders, chigari_folder, models):

    while True:
        emergency_detected = False
        for folder in folders:
            emergency_count = process_folder(folder, models['emergency'], emergency_classes)
            print(f"Emergency vehicle count at {folder}: {emergency_count}")  # Print emergency count

            if emergency_count > 0:
                emergency_detected = True
                print(f"Emergency detected at {folder}. Granting immediate priority.")
                time.sleep(30)
                break

        if emergency_detected:
            continue

        for folder in folders:
            vehicle_count = process_folder(folder, models['vehicle'], vehicle_classes)
            print(f"Vehicle count at {folder}: {vehicle_count}")

            green_time = calculate_green_time(vehicle_count)
            print(f"Green time for {folder}: {green_time} seconds.")
            time.sleep(green_time)

            chigari_count = process_folder(chigari_folder, models['chigari'], vehicle_classes)
            print(f"Chigari vehicle count: {chigari_count}")

            if chigari_count > 0:
                chigari_green_time = calculate_green_time(chigari_count)
                print(f"Chigari priority detected. Granting {chigari_green_time} seconds green time.")
                time.sleep(chigari_green_time)



folders = ["/Users/starkz/PycharmProjects/Yolo_object_detection/Project/Traffic Images/side_1", "/Users/starkz/PycharmProjects/Yolo_object_detection/Project/Traffic Images/side_2", "/Users/starkz/PycharmProjects/Yolo_object_detection/Project/Traffic Images/side_3", "/Users/starkz/PycharmProjects/Yolo_object_detection/Project/Traffic Images/side_4"]
chigari_folder = "/Users/starkz/PycharmProjects/Yolo_object_detection/Project/Traffic Images/chigari"

models = {
    'vehicle': vehicle_model,
    'chigari': chigari_model,
    'emergency': emergency_model
}

traffic_signal_logic(folders, chigari_folder, models)
