import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Initialize Matplotlib figures
fig = plt.figure(figsize=(10, 5))
ax3d = fig.add_subplot(121, projection='3d')
ax_y = fig.add_subplot(122)

# Pose connections
connections = list(mp_pose.POSE_CONNECTIONS)

y_values = []  # Store head Y-axis position
time_steps = []  # Store time steps
squat_count = 0
squat_active = False
squat_threshold = 0.02  # Gradient threshold for squat detection

def update_plot(pose_landmarks):
    global squat_count, squat_active
    ax3d.clear()
    ax_y.clear()
    
    # Set 3D plot view
    ax3d.view_init(elev=90, azim=-90)
    ax3d.set_xlim(-1, 1)
    ax3d.set_ylim(-1, 1)
    ax3d.set_zlim(-1, 1)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_box_aspect([1, 1, 0.3])
    
    # Extract landmark coordinates
    xs, ys, zs = [], [], []
    for landmark in pose_landmarks.landmark:
        xs.append(landmark.x - 0.5)  # Centering X
        ys.append(-landmark.y + 0.5) # Flip Y for correct visualization
        zs.append(-landmark.z)       # MediaPipe has Z in negative
    
    # Plot pose landmarks
    ax3d.scatter(xs, ys, zs, c='red')
    for connection in connections:
        start, end = connection
        ax3d.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], 'b')
    
    # Track Y-axis head movement
    head_y = ys[mp_pose.PoseLandmark.NOSE.value]  # Y position of nose
    y_values.append(head_y)
    time_steps.append(len(time_steps))
    ax_y.plot(time_steps, y_values, 'g')
    ax_y.set_title("Head Y-Axis Tracking")
    ax_y.set_xlabel("Frame")
    ax_y.set_ylabel("Y Position")
    
    # Squat detection based on head movement gradient
    if len(y_values) > 2:
        gradient = np.gradient(y_values[-2:])
        if gradient[-1] < -squat_threshold:  # Detect downward sharp movement
            if not squat_active:
                squat_count += 1
                squat_active = True
        elif gradient[-1] > squat_threshold:  # Reset when standing up
            squat_active = False
    
    ax_y.text(0.05, 0.9, f"Squats: {squat_count}", transform=ax_y.transAxes, fontsize=12, color='red')
    
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