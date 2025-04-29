#!/usr/bin/env python
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from flir_image_extractor import FlirImageExtractor  # Assuming both files are in same directory
from flir import Flir

class FlirThermalProcessor:
    def __init__(self, camera_url, exiftool_path="exiftool"):
        self.camera_url = camera_url
        self.exiftool_path = exiftool_path
        self.fie = FlirImageExtractor(exiftool_path=exiftool_path)
        
        # Initialize FLIR camera connection
        self.flir = Flir(baseURL=camera_url)
        self.flir.login()
    
    def capture_images(self, output_dir="output"):
        """Capture RGB and thermal images without overlay"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Set camera settings
        self.flir.showOverlay(False)  # Disable overlay
        self.flir.setAutoTemperatureRange()  # Use auto scaling
        
        # Capture RGB image
        self.flir.setVisualMode()
        vis_filename = os.path.join(output_dir, f"vis_{timestamp}.jpg")
        self.flir.getSnapshot(vis_filename)
        
        # Capture thermal image (IR only)
        self.flir.setIRMode()
        thermal_filename = os.path.join(output_dir, f"thermal_{timestamp}.jpg")
        self.flir.getSnapshot(thermal_filename)
        
        return vis_filename, thermal_filename
    
    def process_images(self, thermal_filename):
        """Process thermal image to extract temperature data"""
        self.fie.process_image(thermal_filename)
        
        # Get the thermal data in Celsius
        thermal_data = self.fie.get_thermal_np()
        
        return thermal_data
    
    def plot_results(self, RGB_filename, thermal_data):
        """Plot both RGB and thermal images with temperature scale"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot RGB image
        RGB_img = plt.imread(RGB_filename)
        ax1.imshow(RGB_img)
        ax1.set_title('RGB Image')
        ax1.axis('off')
        
        # Plot thermal data
        thermal_plot = ax2.imshow(thermal_data, cmap='hot')
        ax2.set_title('Thermal Data (°C)')
        ax2.axis('off')
        
        # Add colorbar for thermal image
        cbar = fig.colorbar(thermal_plot, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Temperature (°C)')
        
        plt.tight_layout()
        plt.show()
    
    def save_temperature_data(self, thermal_data, output_dir="output"):
        """Save temperature data as numpy array"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_filename = os.path.join(output_dir, f"temperature_{timestamp}.npy")
        np.save(temp_filename, thermal_data)
        print(f"Temperature data saved to {temp_filename}")
        return temp_filename

# Example usage
if __name__ == "__main__":
    # Configuration - change these to match your setup
    CAMERA_URL = "http://192.168.0.10/"  # Your FLIR camera URL
    EXIFTOOL_PATH = "C:/Program Files (x86)/ExifTool/exiftool.exe"  # Path to exiftool if not in PATH
    
    # Create processor instance
    processor = FlirThermalProcessor(camera_url=CAMERA_URL, exiftool_path=EXIFTOOL_PATH)
    
    # 1. Capture images
    vis_img, thermal_img = processor.capture_images()
    
    # 2. Process thermal image to get temperature data
    thermal_data = processor.process_images(thermal_img)
    
    # 3. Plot results
    processor.plot_results(vis_img, thermal_data)
    
    # 4. Save temperature data
    processor.save_temperature_data(thermal_data)