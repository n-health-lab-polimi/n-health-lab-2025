import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import time

# Matplotlib live plot
plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))
xdata, ydata = [], []
line, = ax.plot([], [], label='Pulse Waveform')
text_hr = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='red', fontsize=12)
text_spo2 = ax.text(0.02, 0.90, '', transform=ax.transAxes, color='blue', fontsize=12)

ax.set_ylim(0, 128)
ax.set_ylabel("Waveform")
ax.set_xlabel("Time")
plt.title("rPPG")

# Load the pre-trained face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



target_fps = 20.0
frame_interval = 1.0 / target_fps  # 0.05 seconds


# Start the video capture (use webcam)
cap = cv2.VideoCapture(0)

while True:
    loop_start = time.time()

    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 3,minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Region of interest (ROI) for forehead
        x_roi = x+int(0.2*w)
        w_roi = int(0.6*w)
        y_roi = y
        h_roi = int(0.3*h)
        roi = frame[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi,:]  # Region around the mouth

        # Draw rectangle around the smile
        cv2.rectangle(frame, (x_roi, y_roi), (x_roi+w_roi, y_roi + h_roi), (0, 0, 255), 2)

    xdata.append(loop_start)
    green_channel = roi[:,:,1]
    signal = np.mean(green_channel.flatten())
    ydata.append(signal)
    
    cutoff = loop_start - 10
    xdata = [t for t in xdata if t >= cutoff]
    ydata = ydata[-len(xdata):]


    # Display the resulting frame
    cv2.imshow("rPPG", frame)
    
    line.set_data(xdata, ydata)
    ax.set_xlim(cutoff, loop_start)

    fig.canvas.draw()
    fig.canvas.flush_events()
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Enforce frame rate
    elapsed = time.time() - loop_start
    time_to_wait = frame_interval - elapsed
    if time_to_wait > 0:
        time.sleep(time_to_wait)

cap.release()
cv2.destroyAllWindows()
