import cv2
import numpy as np

def calculate_distance(image1, image2, box1, box2, focal_length, known_width):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate center points of the bounding boxes
    center_x1 = x1 + w1 // 2
    center_y1 = y1 + h1 // 2
    center_x2 = x2 + w2 // 2
    center_y2 = y2 + h2 // 2

    # Calculate distance between centers of the bounding boxes
    distance = (known_width * focal_length) / abs(center_x1 - center_x2)
    return round(distance, 2)

image1 = cv2.imread("marker1.jpeg")
image2 = cv2.imread("marker2.jpeg")

box1 = [100, 200, 50, 1]
box2 = [200, 300, 100, 1]

focal_length = 24
known_width = 150

distance = calculate_distance(image1, image2, box1, box2, focal_length, known_width)
print("Distance: {} mm".format(distance))
