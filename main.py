import cv2
import time
import numpy as np
import torch
import threading
from playsound import playsound  # Using playsound for simple audio playback

from models.ssd_detector import SSDDetector
from models.dvt_classifier import DynamicVisionTransformer
from utils.camera_utils import Camera
from utils.audio_utils import AudioFeedback
from utils.object_tracker import ObjectTracker
from config import DETECTION_FREQUENCY, FRAME_WIDTH, FRAME_HEIGHT

# Load audio files for proximity alerts
VERY_CLOSE_SOUND = "audio/obstacle_very_close.mp3"
NEAR_SOUND = "audio/obstacle_near.mp3"
FAR_SOUND = "audio/obstacle_far.mp3"

def play_proximity_alert(distance):
    """
    Plays different warning sounds based on object proximity.
    - distance < 1.0m -> Very Close Alert
    - 1.0m <= distance < 3.0m -> Near Alert
    - distance >= 3.0m -> Far Alert (optional)
    """
    try:
        if distance < 1.0:
            playsound(VERY_CLOSE_SOUND, block=False)
        elif distance < 3.0:
            playsound(NEAR_SOUND, block=False)
        else:
            playsound(FAR_SOUND, block=False)
    except Exception as e:
        print(f"Error playing sound: {e}")

class EchoStride:
    def __init__(self):
        print("Initializing EchoStride...")
        print("Loading models... This may take a few moments.")
        
        # Initialize components
        self.camera = Camera()
        self.audio = AudioFeedback()
        self.detector = SSDDetector()
        self.transformer = DynamicVisionTransformer()
        self.tracker = ObjectTracker()
        
        # System state
        self.running = False
        self.last_detection_time = 0
        
        print("EchoStride initialized and ready!")
        self.audio.announce_system_status("Echo Stride ready")
        
    def process_frame(self, frame):
        """Process a single frame from the camera"""
        current_time = time.time()
        
        # Perform detection at specified intervals
        if current_time - self.last_detection_time >= DETECTION_FREQUENCY:
            # Detect objects using SSD
            detections = self.detector.detect(frame)
            
            # Track and enhance detections with positional info
            enhanced_detections = self.tracker.update(detections, FRAME_WIDTH, FRAME_HEIGHT)
            
            # Announce detected objects
            self.audio.announce_objects(enhanced_detections, FRAME_WIDTH, FRAME_HEIGHT)

            # Play proximity alerts
            for label, confidence, box, (position, distance) in enhanced_detections:
                threading.Thread(target=play_proximity_alert, args=(distance,), daemon=True).start()
            
            self.last_detection_time = current_time
            
            # Visualize detections (for debugging purposes)
            for label, confidence, box, (position, distance) in enhanced_detections:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        return frame
            
    def run(self):
        """Main loop for the EchoStride system"""
        self.running = True
        self.audio.announce_system_status("Starting object detection")
        
        try:
            while self.running:
                # Get frame from camera
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                    
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Display frame (for development/debugging)
                cv2.imshow("EchoStride", processed_frame)
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Stopping EchoStride...")
        finally:
            self.running = False
            self.camera.release()
            cv2.destroyAllWindows()
            self.audio.announce_system_status("Echo Stride shutting down")
            
    def stop(self):
        """Stop the system"""
        self.running = False

if __name__ == "__main__":
    # Create and run the EchoStride system
    echo_stride = EchoStride()
    echo_stride.run()
