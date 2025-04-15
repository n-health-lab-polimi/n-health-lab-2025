import threading
import queue
import time
import datetime
import serial

class CMS50D:
    def __init__(self, port, baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connection = None
        self.realtime_streaming = False
        self.keepalive_interval = datetime.timedelta(seconds=5)
        self.keepalive_timestamp = datetime.datetime.now()
        #self.data_queue = queue.Queue(maxsize=10)  # Limit queue size to avoid memory overload
        self.data_queue = queue.Queue(maxsize=10)  # Limit queue size to avoid memory overload
        self.thread = None  # Thread for background data collection

    def connect(self):
        self.connection = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            xonxoff=1
        )

    def disconnect(self):
        if self.connection and self.connection.is_open:
            self.connection.close()

    def send_command(self, command):
        def encode_package(cmd):
            package_type = 0x7D
            data = [cmd] + [0x00] * 6
            high_byte = 0x80
            for i in range(len(data)):
                high_byte |= (data[i] & 0x80) >> (7 - i)
                data[i] |= 0x80  # Set sync bit
            package_type &= 0x7F  # Clear sync bit
            return [package_type, high_byte] + data

        package = encode_package(command)
        self.connection.write(bytes(package))
        self.connection.flush()


    def send_keepalive(self):
        now = datetime.datetime.now()
        if now - self.keepalive_timestamp > self.keepalive_interval:
            self.send_command(0xaf)  # keepalive
            self.keepalive_timestamp = now
    
    def start_live_acquisition(self):
        self.connection.reset_input_buffer()
        self.send_command(0xA1)  # Start real-time data
        self.realtime_streaming = True

        # Start a background thread for data collection
        self.thread = threading.Thread(target=self._collect_data)
        self.thread.daemon = True  # Daemonize the thread so it exits with the program
        self.thread.start()

    def stop_live_acquisition(self):
        self.send_command(0xA2)  # Stop real-time data
        self.realtime_streaming = False

    def _collect_data(self):
        while self.realtime_streaming:
            #print("Collecting data...")
            packet = self._read_packet()
            if packet:
                package_type, data = self._decode_packet(packet)
                if package_type == 0x01 and len(data) == 7:
                    # Parse fields and put the data in the queue
                    signal_strength = data[0] & 0x0F
                    pulse_beep = (data[0] & 0x40) >> 6
                    probe_error = (data[0] & 0x80) >> 7
                    pulse_waveform = data[1] & 0x7F
                    pulse_rate = data[3]
                    spO2 = data[4]

                    # Only put the latest data into the queue
                    if not self.data_queue.full():
                        self.data_queue.put({
                            "timestamp": datetime.datetime.now(),
                            "pulse_rate": None if pulse_rate == 0xFF else pulse_rate,
                            "spO2": None if spO2 == 0x7F else spO2,
                            "waveform": pulse_waveform,
                            "signal_strength": signal_strength,
                            "pulse_beep": pulse_beep,
                            "probe_error": probe_error
                        })
                    #else:
                        #print("Queue is full, discarding data.")
                #else:
                    #print("No data received. Check device connection or timeouts.")
            #time.sleep(0.01)  # Add small delay to help stabilize the connection

    def _read_packet(self):
        while True:
            self.send_keepalive()
            byte = self.connection.read()
            if not byte:
                print("Serial timeout or no data received.")
                return None
            #print(f"Reading byte: {byte}")
            if not byte:
                return None
            if not (byte[0] & 0x80):
                packet = byte + self.connection.read(8)
                if len(packet) == 9:
                    return list(packet)

    def _decode_packet(self, packet):
        package_type = packet[0]
        high_byte = packet[1]
        data = list(packet[2:])
        for i in range(len(data)):
            data[i] = (data[i] & 0x7F) | ((high_byte << (7 - i)) & 0x80)
        return package_type, data

    def get_latest_data(self):
        """
        Fetch the latest data from the queue.
        Returns None if no data is available.
        """
        try:
            return self.data_queue.get_nowait()  # Non-blocking, fetch data from queue
        except queue.Empty:
            return None
