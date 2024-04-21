import cv2
import numpy as np
import os

# Function to calculate SSD (Sum of Squared Differences)
def ssd(image1, image2):
    diff = image1 - image2
    return np.sum(diff ** 2)

# Create a directory to save results
save_dir = 'results'
os.makedirs(save_dir, exist_ok=True)

# Load the video
video_path = 'CV Video.mp4'  # Replace 'your_video.mp4' with the path to your video
cap = cv2.VideoCapture(video_path)

# Capture a frame from the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = frame_count // fps

# Extract frames from the video
dataset = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    dataset.append(frame)

# Select a frame from the dataset
selected_frame = dataset[75]

# Select a region of interest (ROI) from the selected frame
roi = cv2.selectROI("Select ROI", selected_frame)
cv2.destroyAllWindows()

# Crop the ROI from the selected frame
roi_x, roi_y, roi_width, roi_height = roi
reference_roi = selected_frame[int(roi_y):int(roi_y + roi_height), int(roi_x):int(roi_x + roi_width)]

# Save the selected ROI
cv2.imwrite(os.path.join(save_dir, 'selected_roi.jpg'), reference_roi)

# Initialize lists to store matched frames and their corresponding SSD scores
matched_frames = []
ssd_scores = []

# Compare the reference ROI with randomly picked frames from the dataset
for i in range(10):
    # Pick a random frame from the dataset
    random_frame_index = np.random.randint(0, len(dataset))
    random_frame = dataset[random_frame_index]
    
    # Crop ROI from the random frame
    random_roi = random_frame[int(roi_y):int(roi_y + roi_height), int(roi_x):int(roi_x + roi_width)]
    
    # Calculate SSD score between reference ROI and random ROI
    score = ssd(reference_roi, random_roi)
    
    # Store the matched frame and its SSD score
    matched_frames.append(random_frame)
    ssd_scores.append(score)

    # Write the SSD score on the matched frame
    cv2.putText(random_frame, f"SSD Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the matched frame
    cv2.imwrite(os.path.join(save_dir, f'matched_frame_{i+1}_ssd_{score}.jpg'), random_frame)

# Display the selected ROI
cv2.imshow("Selected ROI", reference_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the matched frames with their SSD scores
for i in range(10):
    cv2.imshow(f"Matched Frame {i+1} - SSD Score: {ssd_scores[i]}", matched_frames[i])
    cv2.waitKey(0)

# Release the video capture
cap.release()
cv2.destroyAllWindows()