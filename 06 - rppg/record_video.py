import cv2
import time

# Parameters
filename = "output.avi"
target_fps = 20.0
frame_interval = 1.0 / target_fps  # 0.05 seconds
duration = 30  # seconds
frame_width = 640
frame_height = 480

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(filename, fourcc, target_fps, (frame_width, frame_height))

print("Recording at 20 FPS...")

start_time = time.time()
frame_count = 0

while time.time() - start_time < duration:
    loop_start = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed.")
        break

    out.write(frame)
    frame_count += 1

    # Optional display
    # cv2.imshow('Recording', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Enforce frame rate
    elapsed = time.time() - loop_start
    time_to_wait = frame_interval - elapsed
    if time_to_wait > 0:
        time.sleep(time_to_wait)

# Report
total_time = time.time() - start_time
print(f"Done. Captured {frame_count} frames in {total_time:.2f} seconds.")
print(f"Actual FPS: {frame_count / total_time:.2f}")

cap.release()
out.release()
cv2.destroyAllWindows()
