import streamlit as st
import time
from pathlib import Path
import os
from ultralytics import YOLO
import numpy as np
import cv2
from collections import Counter

DEFAULT_GREEN_TIME = 5
MAX_GREEN_TIME = 100
RED_AMBER_TIME = 2
VEHICLE_GREEN_MULTIPLIER = 2
CHIGARI_GREEN_TIME = 10
MIN_GREEN_TIME = 15
LOW_TRAFFIC_GREEN_TIME = 10

IMAGE_FOLDER_PATH = Path(r"/Users/starkz/PycharmProjects/Yolo_object_detection/Project/Traffic Images/")
SIDES = ["side_1", "side_2", "side_3", "side_4"]
CHIGARI_FOLDER = "chigari"


def detect_chagari(images):
    model = YOLO("/Users/starkz/PycharmProjects/Yolo_object_detection/Test Yolo/Yolo_weights/Chigari.pt")

    results = model(images)

    class_names = model.names
    detections = results[0].boxes.cls

    detected_classes = [class_names[int(cls_idx)] for cls_idx in detections]

    class_counts = Counter(detected_classes)

    total_count = 0
    for obj_class, count in class_counts.items():
        st.write(f"{obj_class}: {count}")
        total_count += count

    return total_count

def detect_vehicles(images):
    model = YOLO("/Users/starkz/PycharmProjects/Yolo_object_detection/Test Yolo/Yolo_weights/yolov10s.pt")

    img = cv2.imread(images[0])
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

def calculate_green_time(vehicle_count):
    if vehicle_count == 0:
        return 0
    elif vehicle_count <= 5:
        return LOW_TRAFFIC_GREEN_TIME
    else:
        green_time = 1 + (vehicle_count-2) * 1
        return min(green_time, MAX_GREEN_TIME)

def display_timer(duration, icon, message=None):
    for remaining in range(duration, 0, -1):
        st.write(f"{icon} {message or ''} Timer: {remaining} seconds remaining")
        time.sleep(1)

def traffic_management():
    st.title("Intelligent Traffic Management System")
    st.subheader("Dynamic Green Time Allocation with Priority Handling")
    green_icon = "🟢"
    red_icon = "🔴"
    amber_icon = "🟡"

    side_image_tracker = {side: 0 for side in SIDES}
    chigari_image_tracker = 0

    while True:
        for i, current_side in enumerate(SIDES):
            side_folder = IMAGE_FOLDER_PATH / current_side
            st.write(f"Processing traffic for {current_side}")

            if not side_folder.exists():
                st.write(f"Error: Folder {side_folder} does not exist.")
                continue

            images = list(side_folder.glob("*.jpg")) + list(side_folder.glob("*.png"))
            images.sort(key=lambda img: os.path.getmtime(img), reverse=True)

            vehicle_count = 0

            if side_image_tracker[current_side] >= len(images):
                st.write(f"{red_icon} No more images to process for {current_side}. Moving to next side.")
                st.write(f"{green_icon} {current_side}: No traffic detected. Default green time applied.")
                green_time = DEFAULT_GREEN_TIME
            else:
                if not images:
                    st.write(f"No images found for {current_side}. Default green time applied.")
                    vehicle_count = 0
                else:
                    current_image = images[side_image_tracker[current_side]]
                    vehicle_count = detect_vehicles([current_image])  # Simulate vehicle detection
                    side_image_tracker[current_side] += 1  # Move to the next image

                green_time = calculate_green_time(vehicle_count)

            st.write(f"{green_icon} {current_side}: Traffic Count = {vehicle_count}, Green Time = {green_time} seconds")

            display_timer(green_time, green_icon, message=f"{current_side} Green")

            if green_time == 5:
                if i < len(SIDES) - 1:
                    next_side = SIDES[i + 1]
                    next_side_folder = IMAGE_FOLDER_PATH / next_side
                    next_side_images = list(next_side_folder.glob("*.jpg")) + list(next_side_folder.glob("*.png"))
                    next_side_images.sort(key=lambda img: os.path.getmtime(img), reverse=True)

                    if not next_side_images:
                        next_green_time = DEFAULT_GREEN_TIME
                    else:
                        # Simulate traffic count for the next side
                        next_vehicle_count = detect_vehicles([next_side_images[0]])
                        next_green_time = calculate_green_time(next_vehicle_count)

                    # Display the next side's green time countdown while the current side is still in red/amber
                    st.write(f"{green_icon} Next Side: {next_side}, Green Time = {next_green_time} seconds")
                    display_timer(next_green_time, green_icon, message=f"Next side {next_side} Green Time")

            st.write(f"{red_icon}{amber_icon} Red and Amber Time for {current_side}: {RED_AMBER_TIME} seconds")
            display_timer(RED_AMBER_TIME, amber_icon, message=f"{current_side} Red/Amber")

            chigari_folder = IMAGE_FOLDER_PATH / CHIGARI_FOLDER
            if not chigari_folder.exists():
                st.write(f"Chigari folder {chigari_folder} does not exist. Skipping Chigari simulation.")
                continue

            chigari_images = list(chigari_folder.glob("*.jpg")) + list(chigari_folder.glob("*.png"))
            chigari_images.sort(key=lambda img: os.path.getmtime(img), reverse=True)

            if chigari_image_tracker >= len(chigari_images):
                st.write(f"{red_icon} No more images to process for Chigari.")
            else:
                current_image = chigari_images[chigari_image_tracker]
                chigari_image_tracker += 1

                st.write(f"{red_icon} Chigari Bus Detected! Halting all sides.")
                for side in SIDES:
                    st.write(f"{red_icon} {side}: STOP")

                st.write(f"{green_icon} Chigari Lane: Traffic Count = 1, Green Time = {CHIGARI_GREEN_TIME} seconds")
                display_timer(CHIGARI_GREEN_TIME, green_icon, message="Chigari Lane Green")

            # Break the loop if all images are processed
            if all(side_image_tracker[side] >= len(list((IMAGE_FOLDER_PATH / side).glob("*.jpg")) +
                                                    list((IMAGE_FOLDER_PATH / side).glob("*.png"))) for side in SIDES) \
                    and chigari_image_tracker >= len(chigari_images):
                st.write("All images processed. Simulation complete.")
                break

# Run the application
traffic_management()
