import cv2
import numpy as np
from ultralytics import YOLO
from vidgear.gears import CamGear
import cvzone
from polym import PolylineManager  # Ensure this imports the class correctly

# Initialize the video stream
stream = CamGear(source='https://www.youtube.com/watch?v=_TusTf0iZQU', stream_mode=True, logging=True).start()

# Load COCO class names
with open("coco.txt", "r") as f:
    class_names = f.read().splitlines()

# Load the YOLOv8 model
model = YOLO("yolo11s.pt")

# Create a PolylineManager instance
polyline_manager = PolylineManager()

# Set up the OpenCV window
cv2.namedWindow('RGB')

# Mouse callback to get mouse movements
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Add point on left mouse click
        polyline_manager.add_point((x, y))

# Set the mouse callback function
cv2.setMouseCallback('RGB', RGB)
count = 0
going_up = {}
going_down = {}
gnu=[]
gnd=[]
while True:
    # Read a frame from the video stream
    frame = stream.read()
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True, classes=[2])

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score

        # Draw boxes and labels on the frame
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = class_names[class_id]
            x1, y1, x2, y2 = box
            
            # Calculate the center of the bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            
            if polyline_manager.point_polygon_test((cx, cy), 'area1'):
                going_up[track_id] = (cx, cy)
            if track_id in going_up:
               if polyline_manager.point_polygon_test((cx, cy), 'area2'): 
                  cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                  cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                  cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                  if gnu.count(track_id)==0:
                     gnu.append(track_id)
            if polyline_manager.point_polygon_test((cx, cy), 'area2'):
                going_down[track_id] = (cx, cy)
            if track_id in going_down:
               if polyline_manager.point_polygon_test((cx, cy), 'area1'): 
                  cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                  cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                  cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                  if gnd.count(track_id)==0:
                     gnd.append(track_id)
                
    godown=len(gnd)       
    goup=len(gnu)
    cvzone.putTextRect(frame, f'GoDown:-{godown}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'GoUp:-{goup}', (50, 160), 2, 2)

    # Draw polylines and points on the frame
    frame = polyline_manager.draw_polylines(frame)

    # Display the frame
    cv2.imshow("RGB", frame)

    # Handle key events for polyline management
    if not polyline_manager.handle_key_events():
        break

# Release the video capture object and close the display window
stream.stop()
cv2.destroyAllWindows()
