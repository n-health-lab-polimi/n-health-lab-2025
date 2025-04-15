import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cms50d import CMS50D

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

try:
    while True:
        data = monitor.get_latest_data()  # Get data from the queue
        #print(data)
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

        line.set_data(xdata, ydata)
        ax.set_xlim(cutoff, now)
        text_hr.set_text(f"HR: {data['pulse_rate']} bpm")
        text_spo2.set_text(f"SpO2: {data['spO2']}%")

        fig.canvas.draw()
        fig.canvas.flush_events()
        #time.sleep(0.005)  # Reduced sleep for more frequent updates

except KeyboardInterrupt:
    print("Interrupted by user")
    monitor.stop_live_acquisition()
    monitor.disconnect()

finally:
    monitor.stop_live_acquisition()
    monitor.disconnect()
    plt.ioff()
    plt.show()
