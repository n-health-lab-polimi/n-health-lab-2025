import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Initialize Matplotlib figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define pose connections
connections = list(mp_pose.POSE_CONNECTIONS)

def update_plot(pose_landmarks):
    ax.clear()
    
    # Set view to look at the XY plane (top-down)
    ax.view_init(elev=90, azim=-90)  
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set aspect ratio for better visualization
    ax.set_box_aspect([1, 1, 0.3])  

    # Extract landmark coordinates
    xs, ys, zs = [], [], []
    for landmark in pose_landmarks.landmark:
        xs.append(landmark.x - 0.5)  # Centering X
        ys.append(-landmark.y + 0.5) # Flip Y for correct visualization
        zs.append(-landmark.z)       # MediaPipe has Z in negative

    # Plot pose landmarks
    ax.scatter(xs, ys, zs, c='red')

    # Draw pose connections
    for connection in connections:
        start, end = connection
        ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], 'b')

    plt.draw()
    plt.pause(0.01)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        update_plot(results.pose_landmarks)

    cv2.imshow("Live Pose Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.close()
