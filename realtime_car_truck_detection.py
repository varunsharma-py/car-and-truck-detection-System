import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam, try 1 or 2 for external webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Run YOLOv8 inference
    results = model(frame)
    
    # Draw bounding boxes for cars and trucks
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f'{model.names[cls]} {conf:.2f}'
            if model.names[cls] in ['car', 'truck']:  # Detect cars and trucks
                color = (0, 255, 0) if model.names[cls] == 'car' else (0, 0, 255)  # Green for cars, red for trucks
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Display the frame
    cv2.imshow('Real-Time Car and Truck Detection', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()