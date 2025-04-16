import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import time

# Matplotlib live plot
plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))
xdata, ydata, ydata_filtered = [], [], []
line, = ax.plot([], [], label='Pulse Waveform')
text_hr = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='red', fontsize=12)

ax.set_ylim(0, 128)
ax.set_ylabel("Waveform")
ax.set_xlabel("Time")
plt.title("rPPG")

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Bandpass filter
def bandpass_filter(signal, fs=30, lowcut=0.7, highcut=4.0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    if not (0 < low < high < 1.0):
        print(f"[WARN] Invalid filter range: low={low:.2f}, high={high:.2f} — skipping filter")
        return signal  # fallback to raw signal

    try:
        b, a = butter(4, [low, high], btype="band")
        padlen = 3 * max(len(a), len(b))
        if len(signal) > padlen:
            return filtfilt(b, a, signal)
        else:
            return signal  # too short for filter
    except Exception as e:
        print(f"[ERROR] Filter failed: {e}")
        return signal

# HR estimation functions
def estimate_hr_FFT(waveform, sampling_rate):
    fft_result = np.fft.fft(waveform)
    fft_freqs = np.fft.fftfreq(len(waveform), 1/sampling_rate)
    fft_magnitude = np.abs(fft_result)
    peak_index = np.argmax(fft_magnitude[1:]) + 1
    peak_freq = abs(fft_freqs[peak_index])
    return 60 * peak_freq

def estimate_hr_peak(waveform, sampling_rate):
    peaks, _ = find_peaks(waveform)
    if len(peaks) < 2:
        return 0
    peak_times = np.diff(peaks) / sampling_rate
    avg_peak_interval = np.mean(peak_times)
    return 60 / avg_peak_interval

last_estimation_time = 0
estimation_interval = 1.0  # seconds

hr_fft = 0
hr_peak = 0

# Start the webcam
cap = cv2.VideoCapture(0)
target_fps = 10.0
frame_interval = 1.0 / target_fps

while True:
    loop_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Face and ROI rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        x_roi = x + int(0.2 * w)
        w_roi = int(0.6 * w)
        y_roi = y
        h_roi = int(0.25 * h)
        roi = frame[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi, :]
        cv2.rectangle(frame, (x_roi, y_roi), (x_roi + w_roi, y_roi + h_roi), (0, 0, 255), 2)

        # Green signal
        green_channel = roi[:, :, 1]
        signal = np.mean(green_channel)
        xdata.append(loop_start)
        ydata.append(signal)

        # Keep 20 seconds of data
        cutoff = loop_start - 20
        xdata = [t for t in xdata if t >= cutoff]
        ydata = ydata[-len(xdata):]

        if len(ydata) > 20 and (loop_start - last_estimation_time) >= estimation_interval:
            dt = xdata[-1] - xdata[0]
            sampling_rate = len(xdata) / dt

            filtered_signal = bandpass_filter(ydata, fs=sampling_rate)
            ydata_filtered = filtered_signal

            hr_fft = estimate_hr_FFT(filtered_signal, sampling_rate)
            hr_peak = estimate_hr_peak(filtered_signal, sampling_rate)

            # Update Matplotlib plot
            text_hr.set_text(f"HR (FFT): {hr_fft:.2f} bpm\nHR (Peak): {hr_peak:.2f} bpm")
            line.set_data(xdata, ydata_filtered)
            ax.set_xlim(cutoff, loop_start)
            ax.set_ylim(min(ydata_filtered), max(ydata_filtered))
            fig.canvas.draw()
            fig.canvas.flush_events()

            last_estimation_time = loop_start  # update last estimation timestamp

        # Overlay HR on frame
        cv2.putText(frame, f"HR (FFT): {hr_fft:.1f} bpm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"HR (Peak): {hr_peak:.1f} bpm", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("rPPG", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Enforce frame rate
    elapsed = time.time() - loop_start
    time.sleep(max(0, frame_interval - elapsed))

cap.release()
cv2.destroyAllWindows()
