import cv2
import numpy as np

# Function to calculate SSD (Sum of Squared Differences)
def ssd(image1, image2):
    diff = image1 - image2
    return np.sum(diff ** 2)

# Function to find the best match for the object in the scene
def find_best_match(cropped_region, dataset):
    best_match_index = -1
    best_match_score = float('inf')
    for i, image in enumerate(dataset):
        score = ssd(cropped_region, image)
        if score < best_match_score:
            best_match_score = score
            best_match_index = i
    return best_match_index, best_match_score

# Load your video
video_path = 'CV Video.mp4'
cap = cv2.VideoCapture(video_path)

# Capture a frame from the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = frame_count // fps

# Extract frames
dataset = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    dataset.append(frame)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Select a frame for ROI selection
selected_frame = dataset[0]

# Select ROI using cv2.selectROI()
roi = cv2.selectROI("Frame", selected_frame)

# Crop the region of interest from the selected frame and resize it to match the dataset
roi_x, roi_y, roi_width, roi_height = roi
cropped_region = cv2.resize(selected_frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width], (selected_frame.shape[1], selected_frame.shape[0]))

# Find the best match for the cropped region in the dataset
best_match_index, best_match_score = find_best_match(cropped_region, dataset)

# Save the selected frame, ROI, and best match
cv2.imwrite('selected_frame.jpg', selected_frame)
cv2.imwrite('roi.jpg', cropped_region)
cv2.imwrite('best_match.jpg', dataset[best_match_index])

# Display the best match
cv2.imshow('Cropped Region', cropped_region)
cv2.imshow('Best Match', dataset[best_match_index])
cv2.waitKey(0)

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
