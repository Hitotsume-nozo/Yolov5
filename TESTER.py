import cv2
import numpy as np
from elements.yolo import OBJ_DETECTION

Object_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

Object_colors = list(np.random.rand(80, 3) * 255)
Object_detector = OBJ_DETECTION('weights/yolov5s.pt', Object_classes)

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (frame_width, frame_height))

frame_count = 0
max_frames = 200

if cap.isOpened():
    print("Camera opened! Recording", max_frames, "frames...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if ret:
            objs = Object_detector.detect(frame)

            for obj in objs:
                label = obj['label']
                score = obj['score']
                [(xmin, ymin), (xmax, ymax)] = obj['bbox']
                color = Object_colors[Object_classes.index(label)]
                
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                frame = cv2.putText(frame, f'{label} ({str(score)})', 
                    (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)

            out.write(frame)
            frame_count += 1
            print(f"Frame {frame_count}/{max_frames}", end='\r')
        else:
            break
    
    cap.release()
    out.release()
    print(f"\nDone! Saved to output.avi")
else:
    print("Cannot open camera")
