import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from cms50d import CMS50D 
from scipy.signal import find_peaks

# Function to estimate HR using FFT (Fast Fourier Transform)
def estimate_hr_with_fft(waveform, sampling_rate):
    # Zero padding for improved frequency resolution
    
    # Apply FFT to the signal
    fft_result = np.fft.fft(waveform)
    fft_freqs = np.fft.fftfreq(len(waveform), 1 / sampling_rate)
    
    # Get the magnitude of the FFT result
    fft_magnitude = np.abs(fft_result)

    # Get the index of the maximum peak frequency (ignoring the zero-frequency component)
    peak_index = np.argmax(fft_magnitude[1:]) + 1

    # Estimate heart rate in beats per minute (bpm)
    peak_frequency = abs(fft_freqs[peak_index])  # Frequency in Hz
    hr_bpm = peak_frequency * 60  # Convert Hz to bpm

    return hr_bpm

# Function to estimate HR using peak detection
def estimate_hr_with_peak_detection(waveform, sampling_rate):
    # Convert waveform to numpy array
    waveform = np.array(waveform)

    # Find the peaks in the waveform
    peaks, _ = find_peaks(waveform)

    # Calculate the time between peaks (in seconds)
    peak_times = np.diff(peaks) / sampling_rate  # Time between peaks in seconds

    # Estimate HR by averaging the time between peaks
    avg_peak_interval = np.mean(peak_times) if len(peak_times) > 0 else 1
    hr_bpm = 60 / avg_peak_interval  # Convert to bpm

    return hr_bpm

# Main monitoring loop
monitor = CMS50D(port="COM16")  # Replace with your actual COM port
monitor.connect()
monitor.start_live_acquisition()

# Matplotlib live plot
plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))
xdata, ydata = [], []
line, = ax.plot_date([], [], fmt='-m', label='Pulse Waveform')
text_hr = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='red', fontsize=12)
text_spo2 = ax.text(0.02, 0.90, '', transform=ax.transAxes, color='blue', fontsize=12)

ax.set_ylim(0, 128)
ax.set_ylabel("Waveform")
ax.set_xlabel("Time")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.title("CMS50D Live Data")

last_timestamp = None
sampling_rate = None

try:
    while True:
        data = monitor.get_latest_data()  # Get data from the queue
        if not data:
            continue

        now = data['timestamp']
        waveform = data['waveform']
        xdata.append(now)
        ydata.append(waveform)

        # Keep last 10 seconds of data
        cutoff = now - datetime.timedelta(seconds=10)
        xdata = [t for t in xdata if t >= cutoff]
        ydata = ydata[-len(xdata):]

        dt = (xdata[-1] - xdata[0]).total_seconds()

        if dt > 0:
            sampling_rate = len(xdata) / dt  # In Hz (samples per second)
            print(sampling_rate)

        # Estimate HR using FFT and Peak Detection
        if sampling_rate:
            hr_fft = estimate_hr_with_fft(ydata, sampling_rate)
            hr_peak = estimate_hr_with_peak_detection(ydata, sampling_rate)

            # Display the estimated HR values
            text_hr.set_text(f"HR (FFT): {hr_fft:.2f} bpm\nHR (Peak): {hr_peak:.2f} bpm\nHR (real): {data['pulse_rate']:.2f} bpm")
            text_spo2.set_text(f"SpO2: {data['spO2']}%")

        line.set_data(xdata, ydata)
        ax.set_xlim(cutoff, now)

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)  # Reduced sleep for more frequent updates

except KeyboardInterrupt:
    print("Interrupted by user")
    monitor.stop_live_acquisition()
    monitor.disconnect()

finally:
    monitor.stop_live_acquisition()
    monitor.disconnect()
    plt.ioff()
    plt.show()
