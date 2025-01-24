import streamlit as st
import time
from pathlib import Path
import os
from ultralytics import YOLO
import numpy as np
import cv2

def detect_vehicles(images):
    model = YOLO("/Users/starkz/PycharmProjects/Yolo_object_detection/Test Yolo/Yolo_weights/yolov10s.pt")

    img = cv2.imread(str(images))
    imgreg = img
    res = model.predict(imgreg, stream=True)

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                  "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                  "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
                  "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                  "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
                  "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

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
    st.write("Count =", vehicle_count)

    return vehicle_count

def detect_chigari_buses(chigari_folder):

    model = YOLO("/Users/starkz/PycharmProjects/Yolo_object_detection/Test Yolo/Yolo_weights/Chigari.pt")

    if not chigari_folder.exists():
        st.write(f"Error: Chigari folder {chigari_folder} does not exist.")
        return 0

    chigari_images = list(chigari_folder.glob("*.jpg")) + list(chigari_folder.glob("*.png"))
    chigari_images.sort(key=lambda img: os.path.getmtime(img), reverse=True)  # Sort by recent timestamp

    if not chigari_images:
        st.write("No images found in Chigari folder.")
        return 0

    recent_image = chigari_images[0]
    st.write(f"Processing the most recent Chigari image: {recent_image.name}")

    img = cv2.imread(str(recent_image))
    res = model(img, stream=True)

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                  "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                  "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
                  "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                  "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
                  "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "Chigari"]

    chigari_count = 0
    for r in res:
        boxes = r.boxes
        for box in boxes:
            confi = round(float(box.conf[0] * 100))
            cls = box.cls[0]
            current_class = classNames[int(cls)]

            if current_class == "Chigari" and confi >= 30:
                chigari_count += 1

    st.write(f"Chigari Count = {chigari_count}")
    return chigari_count

def calculate_green_time(vehicle_count):
    if vehicle_count == 0:
        return 0
    elif vehicle_count <= 5:
        return 10
    else:
        green_time = 1 + (vehicle_count - 2) * 1
        return min(green_time, 100)

def display_timer(duration, icon):
    for remaining in range(duration, 0, -1):
        st.write(f"{icon} Timer: {remaining} seconds remaining")
        time.sleep(1)

def traffic_management():
    st.title("Intelligent Traffic Management System")

    IMAGE_FOLDER_PATH = Path("/Users/starkz/PycharmProjects/Yolo_object_detection/Project/Traffic Images/")
    SIDES = ["side_1", "side_2", "side_3", "side_4"]
    CHIGARI_FOLDER = IMAGE_FOLDER_PATH / "chigari"

    side_image_tracker = {side: 0 for side in SIDES}

    while True:
        chigari_count = detect_chigari_buses(CHIGARI_FOLDER)
        if chigari_count > 0:
            st.write("Chigari Bus Detected! Interrupting all sides.")
            st.write(f"Allowing Chigari buses with green time of 10 seconds.")
            display_timer(10, "\U0001F7E2")  # Green timer for Chigari buses
            continue

        for current_side in SIDES:
            side_folder = IMAGE_FOLDER_PATH / current_side

            if not side_folder.exists():
                st.write(f"Error: Folder {side_folder} does not exist.")
                continue

            images = list(side_folder.glob("*.jpg")) + list(side_folder.glob("*.png"))
            images.sort(key=lambda img: os.path.getmtime(img), reverse=True)

            if side_image_tracker[current_side] >= len(images):
                st.write("No more images to process for this side.")
                continue

            current_image = images[side_image_tracker[current_side]]
            vehicle_count = detect_vehicles(current_image)
            side_image_tracker[current_side] += 1

            green_time = calculate_green_time(vehicle_count)

            st.write(f"Current side: {current_side}")
            display_timer(green_time, "\U0001F7E2")

traffic_management()