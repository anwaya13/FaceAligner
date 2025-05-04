import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import os

# Initialize dlib's face detector (HOG-based) and facial landmarks predictor
detector = dlib.get_frontal_face_detector()

# Ensure the shape predictor file is present in the directory
predictor_path = r"C:\Users\athul\Downloads\dumbo_project\shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    raise FileNotFoundError(f"Shape predictor file not found: {predictor_path}")

predictor = dlib.shape_predictor(predictor_path)

# Helper function to convert the image from BGR to RGB (for displaying with matplotlib)
def convert_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to extract facial keypoints
def extract_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    all_keypoints = []
    
    if len(faces) == 0:
        print("No faces detected.")
        return None
    
    for face in faces:
        keypoints = []
        landmarks = predictor(gray, face)
        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            keypoints.append((x, y))
        all_keypoints.append(keypoints)
    return all_keypoints

# Function to draw keypoints on image
def draw_keypoints(image, keypoints):
    if keypoints is None:
        return image
    for point in keypoints:
        cv2.circle(image, tuple(map(int, point)), 2, (0, 255, 0), -1)
    return image

# Function to align face based on the eye positions (pose invariant)
# Function to align face based on the eye positions (pose invariant)
def align_face(image, keypoints):
    left_eye = np.mean(keypoints[36:42], axis=0).astype(int)  # Left eye keypoints (36-41)
    right_eye = np.mean(keypoints[42:48], axis=0).astype(int)  # Right eye keypoints (42-47)
    
    # Calculate angle and distance between the two eyes
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))  # Removed the -180 offset
    
    # Calculate center point between the eyes (ensuring it's an integer tuple)
    eyes_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
    
    # Get the rotation matrix for the given angle
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)
    
    # Rotate the image to align face
    aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
    
    return aligned_image, M



# Function to align keypoints
def transform_keypoints(keypoints, M):
    if M is None:
        return keypoints
    keypoints = np.array(keypoints)
    keypoints = np.c_[keypoints, np.ones(len(keypoints))]  # Add a third dimension for the transformation
    transformed_keypoints = np.dot(M, keypoints.T).T
    return transformed_keypoints

# Function to compare faces using keypoints (Euclidean distance)
def compare_faces(keypoints1, keypoints2):
    if keypoints1 is None or keypoints2 is None:
        return float("inf")  # Return a high distance if comparison is invalid
    assert len(keypoints1) == len(keypoints2), "Key points should be the same length."
    distances = [distance.euclidean(p1, p2) for p1, p2 in zip(keypoints1, keypoints2)]
    return np.mean(distances)

# Display the image with Matplotlib
def show_image(image, title="Image"):
    plt.figure(figsize=(8,8))
    plt.imshow(convert_to_rgb(image))
    plt.title(title)
    plt.axis("off")
    plt.show()

# Function to display keypoints and differences between keypoints of two images
# Helper function to resize images while maintaining aspect ratio
def resize_image(image, target_height):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, target_height))
    return resized_image

# Function to display keypoints and differences between keypoints of two images
def show_differences(image1, image2, keypoints1, keypoints2, same_person):
    # Ensure both images have the same height before concatenating
    height1 = image1.shape[0]
    height2 = image2.shape[0]
    
    # Resize the second image to match the height of the first image (or vice versa)
    if height1 != height2:
        target_height = min(height1, height2)
        image1 = resize_image(image1, target_height)
        image2 = resize_image(image2, target_height)
    
    # Combine both images side by side
    combined_image = np.hstack((image1, image2))
    
    if same_person:
        message = "SAME PERSON"
        color = (0, 255, 0)  # Green for similar faces
    else:
        message = "DIFFERENT PERSON"
        color = (0, 0, 255)  # Red for different faces
    
    # Draw lines between corresponding keypoints if the faces are different
    for i, (kp1, kp2) in enumerate(zip(keypoints1, keypoints2)):
        kp1 = tuple(map(int, kp1))
        kp2 = tuple(map(int, kp2))
        # Adjust the x coordinate of keypoints in image2 to match the combined image
        kp2_adjusted = (kp2[0] + image1.shape[1], kp2[1])  
        
        # Draw a line for different keypoints (only when the person is different)
        if not same_person:
            cv2.line(combined_image, kp1, kp2_adjusted, (255, 0, 0), 1)
    0
    # Add the result message to the combined image
    cv2.putText(combined_image, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    return combined_image



# Load and process images
image1_path = r"C:\Users\athul\Downloads\dumbo_project\face1.jpg"
image2_path = r"C:\Users\athul\Downloads\dumbo_project\face2.jpg"

# Load images using cv2
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Check if images are loaded properly
if image1 is None or image2 is None:
    raise FileNotFoundError(f"One or both images could not be loaded: {image1_path}, {image2_path}")

# Extract keypoints from both images
keypoints1_list = extract_keypoints(image1)
keypoints2_list = extract_keypoints(image2)

if keypoints1_list and keypoints2_list:
    # Use the first detected face for comparison (if multiple faces)
    keypoints1 = keypoints1_list[0]
    keypoints2 = keypoints2_list[0]

    # Align both faces for better comparison
    aligned_face1, M1 = align_face(image1, keypoints1)
    aligned_face2, M2 = align_face(image2, keypoints2)

    # Align the keypoints using the same transformation matrix
    aligned_keypoints1 = transform_keypoints(keypoints1, M1)
    aligned_keypoints2 = transform_keypoints(keypoints2, M2)

    # Compare the two faces using Euclidean distance of the keypoints
    similarity_score = compare_faces(aligned_keypoints1, aligned_keypoints2)
    threshold = 10.0  # You can adjust this threshold based on experiments

    same_person = similarity_score < threshold

    # Display the comparison result
    result_image = show_differences(aligned_face1, aligned_face2, aligned_keypoints1, aligned_keypoints2, same_person)
    
    # Display the final image with the result
    show_image(result_image, "Face Comparison Result")

    print(f"Similarity Score: {similarity_score}")
    if same_person:
        print("Faces are similar!")
    else:
        print("Faces are different!")
else:
    print("Face comparison could not be performed due to missing keypoints.")
