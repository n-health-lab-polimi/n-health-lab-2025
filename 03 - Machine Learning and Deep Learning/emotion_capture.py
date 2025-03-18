import cv2
import os
import time

class EmotionCapture:
    def __init__(self, capture_time=10):
        self.capture_time = capture_time
        self.label = None
        self.capture = False
        self.start_time = None
        self.count = 0
        
        # Create directories for saving images
        os.makedirs('data/emotion/happy', exist_ok=True)
        os.makedirs('data/emotion/sad', exist_ok=True)
        
        # Load Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Start video capture
        self.cap = cv2.VideoCapture(0)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        
        frame_visual = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(200, 200))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_visual, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (64, 64))
            
            if self.capture:
                face_file_path = f"data/emotion/{self.label}/{self.label}_{self.count}.jpg"
                cv2.imwrite(face_file_path, face_resized)
                self.count += 1
        
        return frame_visual, faces

    def run(self):
        while True:
            frame_visual, faces = self.process_frame()
            if frame_visual is None:
                break
            
            cv2.putText(frame_visual, "Press 'h' for Happy, 's' for Sad, 'q' to Quit", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            if self.capture:
                elapsed_time = time.time() - self.start_time
                remaining_time = max(self.capture_time - int(elapsed_time), 0)
                cv2.putText(frame_visual, f"Recording {self.label}: {remaining_time}s left", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                if elapsed_time >= self.capture_time:
                    self.capture = False
                    print("Done")
            
            cv2.imshow('Emotion Capture', frame_visual)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('h') or key == ord('s'):
                self.label = 'happy' if key == ord('h') else 'sad'
                self.capture = True
                self.start_time = time.time()
                self.count = 0
                print(f"Started capturing {self.label}")
            
            if key == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    EmotionCapture().run()
