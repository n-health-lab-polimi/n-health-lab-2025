import time
import matplotlib.pyplot as plt
from witmotion import IMU

# Initialize the IMU sensor
imu = IMU('COM10')

# Lists to store accelerometer and gyroscope data
time_data = []
accel_x, accel_y, accel_z = [], [], []
gyro_x, gyro_y, gyro_z = [], [], []

# Initialize live plotting
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))  # Two subplots: Acceleration & Gyro

# Labels and titles
ax1.set_title("Accelerometer Data (m/s²)")
ax1.set_ylabel("Acceleration")
ax1.legend(["X", "Y", "Z"], loc="upper right")

ax2.set_title("Gyroscope Data (°/s)")
ax2.set_ylabel("Angular Velocity")
ax2.legend(["X", "Y", "Z"], loc="upper right")

ax2.set_xlabel("Time (s)")

# Start time
start_time = time.time()

# Function to get and plot data in real-time
def acquire_data_and_plot():
    while True:
        current_time = time.time() - start_time  # Track elapsed time

        # Get acceleration and gyro data
        a = imu.get_acceleration()  # Returns (ax, ay, az)
        g = imu.get_angular_velocity()  # Returns (gx, gy, gz)

        if a and g:  # Ensure data is valid
            time_data.append(current_time)

            # Append acceleration data
            accel_x.append(a[0])
            accel_y.append(a[1])
            accel_z.append(a[2])

            # Append gyroscope data
            gyro_x.append(g[0])
            gyro_y.append(g[1])
            gyro_z.append(g[2])

            # Limit data points to the last 100 for better visualization
            if len(time_data) > 100:
                time_data.pop(0)
                accel_x.pop(0)
                accel_y.pop(0)
                accel_z.pop(0)
                gyro_x.pop(0)
                gyro_y.pop(0)
                gyro_z.pop(0)

            # Clear old plots
            ax1.clear()
            ax2.clear()

            # Re-plot acceleration data
            ax1.plot(time_data, accel_x, label="X", color="r")
            ax1.plot(time_data, accel_y, label="Y", color="g")
            ax1.plot(time_data, accel_z, label="Z", color="b")
            ax1.set_title("Accelerometer Data (m/s²)")
            ax1.set_ylabel("Acceleration")
            ax1.legend(loc="upper right")

            # Re-plot gyroscope data
            ax2.plot(time_data, gyro_x, label="X", color="r")
            ax2.plot(time_data, gyro_y, label="Y", color="g")
            ax2.plot(time_data, gyro_z, label="Z", color="b")
            ax2.set_title("Gyroscope Data (°/s)")
            ax2.set_ylabel("Angular Velocity")
            ax2.set_xlabel("Time (s)")
            ax2.legend(loc="upper right")

            # Update plot
            plt.draw()
            plt.pause(0.1)  # Pause briefly to allow the plot to update

        # Stop after 60 seconds
        if current_time > 60:
            imu.close()
            break

    plt.ioff()  # Turn off interactive mode
    plt.show()

# Run the data acquisition and plotting function
acquire_data_and_plot()
imu.close()
