cat > BetterYoloUp.py << 'ENDOFSCRIPT'
from flask import Flask, Response, render_template_string, send_file
import cv2
import numpy as np
from elements.yolo import OBJ_DETECTION
from collections import Counter
import json
import threading
import time
import csv
import os
import signal
import sys
from datetime import datetime

app = Flask(__name__)

Object_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush']

Object_colors = list(np.random.rand(80, 3) * 255)
Object_detector = OBJ_DETECTION('weights/yolov5s.pt', Object_classes)
cap = cv2.VideoCapture(0)

# Shared data
frame_data = {
    'objects': [],
    'fps': 0,
    'frame_count': 0,
    'frame_hsv': {},
    'timestamp': ''
}
data_lock = threading.Lock()

# CSV data storage
csv_records = []
csv_lock = threading.Lock()
session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"detection_log_{session_start}.csv"

CSV_HEADERS = [
    'timestamp', 'frame_number', 'fps',
    'object_id', 'label', 'confidence',
    'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax',
    'bbox_width', 'bbox_height',
    'color_1_name', 'color_1_rgb', 'color_1_hsv', 'color_1_pct',
    'color_2_name', 'color_2_rgb', 'color_2_hsv', 'color_2_pct',
    'color_3_name', 'color_3_rgb', 'color_3_hsv', 'color_3_pct',
    'hsv_h_mean', 'hsv_h_std', 'hsv_h_min', 'hsv_h_max',
    'hsv_s_mean', 'hsv_s_std', 'hsv_s_min', 'hsv_s_max',
    'hsv_v_mean', 'hsv_v_std', 'hsv_v_min', 'hsv_v_max',
    'moment_m00', 'moment_m10', 'moment_m01',
    'moment_m20', 'moment_m11', 'moment_m02',
    'moment_mu20', 'moment_mu11', 'moment_mu02',
    'moment_nu20', 'moment_nu11', 'moment_nu02',
    'centroid_x', 'centroid_y',
    'orientation_deg',
    'hu_0', 'hu_1', 'hu_2', 'hu_3', 'hu_4', 'hu_5', 'hu_6',
    'frame_h_mean', 'frame_s_mean', 'frame_v_mean'
]


def export_csv():
    """Export all collected data to CSV"""
    with csv_lock:
        if len(csv_records) == 0:
            print("[INFO] No data to export.")
            return
        
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction='ignore')
            writer.writeheader()
            for record in csv_records:
                writer.writerow(record)
        
        print(f"[EXPORT] Saved {len(csv_records)} records to {filepath}")


def signal_handler(sig, frame):
    """Handle shutdown signals"""
    print("\n[INFO] Shutdown signal received. Exporting CSV...")
    export_csv()
    cap.release()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_dominant_colors(image, k=3):
    """K-means dominant color extraction"""
    if image.size == 0 or image.shape[0] < 2 or image.shape[1] < 2:
        return []
    
    small = cv2.resize(image, (30, 30))
    pixels = small.reshape(-1, 3).astype(np.float32)
    
    k = min(k, len(pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    try:
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        label_counts = Counter(labels.flatten())
        total = sum(label_counts.values())
        
        colors = []
        for idx, count in label_counts.most_common(k):
            bgr = centers[idx].astype(int)
            rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
            hsv_pixel = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            percentage = round(count / total * 100, 1)
            color_name = hsv_to_color_name(hsv_pixel[0], hsv_pixel[1], hsv_pixel[2])
            
            colors.append({
                'rgb': rgb,
                'hsv': (int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])),
                'percentage': percentage,
                'name': color_name
            })
        return colors
    except:
        return []


def hsv_to_color_name(h, s, v):
    """HSV to human-readable color classification"""
    if v < 30:
        return "Black"
    if v < 60 and s < 40:
        return "Dark Gray"
    if s < 25:
        if v > 200:
            return "White"
        elif v > 140:
            return "Light Gray"
        elif v > 80:
            return "Gray"
        else:
            return "Dark Gray"
    
    if h < 8 or h > 170:
        if s > 150:
            return "Red"
        else:
            return "Light Red"
    elif h < 22:
        return "Orange"
    elif h < 35:
        return "Yellow"
    elif h < 50:
        return "Yellow-Green"
    elif h < 78:
        return "Green"
    elif h < 100:
        return "Cyan"
    elif h < 130:
        return "Blue"
    elif h < 145:
        return "Purple"
    elif h < 170:
        return "Magenta"
    return "Unknown"


def calculate_full_moments(roi):
    """Calculate comprehensive moments including 0th and 2nd order"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    m = cv2.moments(gray)
    hu = cv2.HuMoments(m)
    
    result = {
        # Raw spatial moments (0th order)
        'm00': round(m['m00'], 4),
        'm10': round(m['m10'], 4),
        'm01': round(m['m01'], 4),
        # Raw spatial moments (2nd order)
        'm20': round(m['m20'], 4),
        'm11': round(m['m11'], 4),
        'm02': round(m['m02'], 4),
        # Central moments (2nd order)
        'mu20': round(m['mu20'], 4),
        'mu11': round(m['mu11'], 4),
        'mu02': round(m['mu02'], 4),
        # Normalized central moments (2nd order)
        'nu20': round(m['nu20'], 8),
        'nu11': round(m['nu11'], 8),
        'nu02': round(m['nu02'], 8),
        # Derived
        'centroid_x': 0,
        'centroid_y': 0,
        'orientation': 0,
        'area': round(m['m00'], 1),
        # Hu moments (log transformed)
        'hu_moments': [round(float(-np.sign(h) * np.log10(abs(h) + 1e-10)), 4) for h in hu.flatten()]
    }
    
    if m['m00'] != 0:
        result['centroid_x'] = round(m['m10'] / m['m00'], 2)
        result['centroid_y'] = round(m['m01'] / m['m00'], 2)
    
    if (m['mu20'] - m['mu02']) != 0:
        result['orientation'] = round(
            0.5 * np.arctan2(2 * m['mu11'], m['mu20'] - m['mu02']) * 180 / np.pi, 2
        )
    
    return result


def get_hsv_stats(roi):
    """Comprehensive HSV statistics"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return {
        'h_mean': round(float(np.mean(hsv[:,:,0])), 2),
        'h_std': round(float(np.std(hsv[:,:,0])), 2),
        'h_min': int(np.min(hsv[:,:,0])),
        'h_max': int(np.max(hsv[:,:,0])),
        'h_median': round(float(np.median(hsv[:,:,0])), 2),
        's_mean': round(float(np.mean(hsv[:,:,1])), 2),
        's_std': round(float(np.std(hsv[:,:,1])), 2),
        's_min': int(np.min(hsv[:,:,1])),
        's_max': int(np.max(hsv[:,:,1])),
        's_median': round(float(np.median(hsv[:,:,1])), 2),
        'v_mean': round(float(np.mean(hsv[:,:,2])), 2),
        'v_std': round(float(np.std(hsv[:,:,2])), 2),
        'v_min': int(np.min(hsv[:,:,2])),
        'v_max': int(np.max(hsv[:,:,2])),
        'v_median': round(float(np.median(hsv[:,:,2])), 2)
    }


def build_csv_record(frame_num, fps, timestamp, obj_idx, obj_data, frame_hsv):
    """Build a single CSV record from detection data"""
    record = {
        'timestamp': timestamp,
        'frame_number': frame_num,
        'fps': fps,
        'object_id': obj_idx,
        'label': obj_data['label'],
        'confidence': obj_data['score'],
        'bbox_xmin': obj_data['bbox'][0],
        'bbox_ymin': obj_data['bbox'][1],
        'bbox_xmax': obj_data['bbox'][2],
        'bbox_ymax': obj_data['bbox'][3],
        'bbox_width': obj_data['bbox'][2] - obj_data['bbox'][0],
        'bbox_height': obj_data['bbox'][3] - obj_data['bbox'][1],
        'frame_h_mean': frame_hsv.get('h_mean', 0),
        'frame_s_mean': frame_hsv.get('s_mean', 0),
        'frame_v_mean': frame_hsv.get('v_mean', 0)
    }
    
    # Colors
    for i in range(3):
        ci = i + 1
        if i < len(obj_data.get('colors', [])):
            c = obj_data['colors'][i]
            record[f'color_{ci}_name'] = c['name']
            record[f'color_{ci}_rgb'] = f"{c['rgb'][0]},{c['rgb'][1]},{c['rgb'][2]}"
            record[f'color_{ci}_hsv'] = f"{c['hsv'][0]},{c['hsv'][1]},{c['hsv'][2]}"
            record[f'color_{ci}_pct'] = c['percentage']
        else:
            record[f'color_{ci}_name'] = ''
            record[f'color_{ci}_rgb'] = ''
            record[f'color_{ci}_hsv'] = ''
            record[f'color_{ci}_pct'] = ''
    
    # HSV stats
    hsv = obj_data.get('hsv', {})
    for ch in ['h', 's', 'v']:
        for stat in ['mean', 'std', 'min', 'max']:
            record[f'hsv_{ch}_{stat}'] = hsv.get(f'{ch}_{stat}', 0)
    
    # Moments
    moments = obj_data.get('moments', {})
    for key in ['m00', 'm10', 'm01', 'm20', 'm11', 'm02',
                'mu20', 'mu11', 'mu02', 'nu20', 'nu11', 'nu02']:
        record[f'moment_{key}'] = moments.get(key, 0)
    
    record['centroid_x'] = moments.get('centroid_x', 0)
    record['centroid_y'] = moments.get('centroid_y', 0)
    record['orientation_deg'] = moments.get('orientation', 0)
    
    hu = moments.get('hu_moments', [0]*7)
    for i in range(7):
        record[f'hu_{i}'] = hu[i] if i < len(hu) else 0
    
    return record


def draw_color_bar(frame, colors, x, y, bar_width=90, bar_height=12):
    """Render color composition bar"""
    current_x = x
    for color_info in colors:
        r, g, b = color_info['rgb']
        width = max(1, int(bar_width * color_info['percentage'] / 100))
        cv2.rectangle(frame, (current_x, y), (current_x + width, y + bar_height), (b, g, r), -1)
        current_x += width
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (200, 200, 200), 1)


def draw_hsv_bars(frame, hsv_stats, x, y):
    """Render HSV channel bars with scales"""
    bar_width = 110
    bar_height = 7
    gap = 12
    
    channels = [
        ('H', hsv_stats['h_mean'], 180, (0, 180, 220), hsv_stats['h_std']),
        ('S', hsv_stats['s_mean'], 255, (0, 200, 0), hsv_stats['s_std']),
        ('V', hsv_stats['v_mean'], 255, (220, 220, 220), hsv_stats['v_std'])
    ]
    
    for label, val, max_val, color, std in channels:
        fill = int(bar_width * val / max_val)
        std_start = int(bar_width * max(0, val - std) / max_val)
        std_end = int(bar_width * min(max_val, val + std) / max_val)
        
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (40, 40, 40), -1)
        cv2.rectangle(frame, (x + std_start, y), (x + std_end, y + bar_height), (60, 60, 60), -1)
        cv2.rectangle(frame, (x, y), (x + fill, y + bar_height), color, -1)
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (100, 100, 100), 1)
        
        cv2.putText(frame, f"{label}:{int(val)}", (x + bar_width + 4, y + bar_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)
        y += gap


def draw_moment_overlay(frame, moments_data, xmin, ymin, xmax, ymax):
    """Draw centroid crosshair and orientation line"""
    if moments_data['area'] <= 0:
        return
    
    cx = xmin + int(moments_data['centroid_x'])
    cy = ymin + int(moments_data['centroid_y'])
    
    # Centroid crosshair
    size = 10
    cv2.line(frame, (cx - size, cy), (cx + size, cy), (0, 220, 220), 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), (0, 220, 220), 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 2, (0, 220, 220), -1, cv2.LINE_AA)
    
    # Orientation line
    angle_rad = moments_data['orientation'] * np.pi / 180
    line_len = min(xmax - xmin, ymax - ymin) // 3
    ex = int(cx + line_len * np.cos(angle_rad))
    ey = int(cy + line_len * np.sin(angle_rad))
    cv2.line(frame, (cx, cy), (ex, ey), (0, 180, 255), 1, cv2.LINE_AA)


def generate_frames():
    """Main detection stream with full overlay"""
    prev_time = time.time()
    fps_buffer = []
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 0.0001)
        prev_time = current_time
        fps_buffer.append(fps)
        if len(fps_buffer) > 30:
            fps_buffer.pop(0)
        avg_fps = sum(fps_buffer) / len(fps_buffer)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        objs = Object_detector.detect(frame)
        full_hsv = get_hsv_stats(frame)
        
        h, w = frame.shape[:2]
        obj_data_list = []
        
        for idx, obj in enumerate(objs):
            label = obj['label']
            score = obj['score']
            [(xmin, ymin), (xmax, ymax)] = obj['bbox']
            color = Object_colors[Object_classes.index(label)]
            
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            
            roi = frame[ymin:ymax, xmin:xmax]
            
            if roi.size > 0 and roi.shape[0] > 1 and roi.shape[1] > 1:
                dominant_colors = get_dominant_colors(roi, k=3)
                hsv_stats = get_hsv_stats(roi)
                moments_data = calculate_full_moments(roi)
                
                obj_entry = {
                    'label': label,
                    'score': score,
                    'bbox': [xmin, ymin, xmax, ymax],
                    'colors': dominant_colors,
                    'hsv': hsv_stats,
                    'moments': moments_data
                }
                obj_data_list.append(obj_entry)
                
                # Store CSV record
                csv_record = build_csv_record(frame_num, round(avg_fps, 2), timestamp, idx, obj_entry, full_hsv)
                with csv_lock:
                    csv_records.append(csv_record)
                
                # --- DRAW BOUNDING BOX ---
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                
                # --- LABEL BAR ---
                label_text = f"{label} {score:.0%}"
                ts = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                cv2.rectangle(frame, (xmin, ymin - 18), (xmin + ts[0] + 6, ymin), color, -1)
                cv2.putText(frame, label_text, (xmin + 3, ymin - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                
                # --- COLOR BAR ---
                if dominant_colors:
                    draw_color_bar(frame, dominant_colors, xmin, ymax + 2)
                    color_labels = " | ".join([f"{c['name']} {c['percentage']:.0f}%" for c in dominant_colors[:3]])
                    cv2.putText(frame, color_labels, (xmin, ymax + 26),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1, cv2.LINE_AA)
                
                # --- HSV BARS ---
                draw_hsv_bars(frame, hsv_stats, xmax + 4, ymin)
                
                # --- MOMENT OVERLAY ---
                draw_moment_overlay(frame, moments_data, xmin, ymin, xmax, ymax)
                
                # --- MOMENT TEXT ---
                m_y = ymin - 22
                cv2.putText(frame, f"M00:{moments_data['m00']:.0f} | Angle:{moments_data['orientation']:.1f}deg",
                           (xmin, m_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(frame, f"M20:{moments_data['m20']:.0f} M02:{moments_data['m02']:.0f} M11:{moments_data['m11']:.0f}",
                           (xmin, m_y - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(frame, f"nu20:{moments_data['nu20']:.6f} nu02:{moments_data['nu02']:.6f}",
                           (xmin, m_y - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 180, 180), 1, cv2.LINE_AA)
        
        # ===== GLOBAL INFO PANEL =====
        panel_w = 260
        panel_h = 175
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        px, py = 16, 24
        cv2.putText(frame, "DETECTION SYSTEM", (px, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.line(frame, (px, py + 4), (px + 155, py + 4), (60, 60, 60), 1)
        
        py += 22
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (px, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {frame_num}", (px + 120, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 0), 1, cv2.LINE_AA)
        
        py += 18
        cv2.putText(frame, f"Objects: {len(objs)}", (px, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        
        with csv_lock:
            cv2.putText(frame, f"CSV Records: {len(csv_records)}", (px + 120, py),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        
        py += 20
        cv2.putText(frame, "Frame HSV Statistics", (px, py),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
        py += 16
        cv2.putText(frame, f"H: {full_hsv['h_mean']:.1f} +/-{full_hsv['h_std']:.1f}  [{full_hsv['h_min']}-{full_hsv['h_max']}]",
                   (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 180, 220), 1, cv2.LINE_AA)
        py += 14
        cv2.putText(frame, f"S: {full_hsv['s_mean']:.1f} +/-{full_hsv['s_std']:.1f}  [{full_hsv['s_min']}-{full_hsv['s_max']}]",
                   (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 200, 0), 1, cv2.LINE_AA)
        py += 14
        cv2.putText(frame, f"V: {full_hsv['v_mean']:.1f} +/-{full_hsv['v_std']:.1f}  [{full_hsv['v_min']}-{full_hsv['v_max']}]",
                   (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1, cv2.LINE_AA)
        
        py += 18
        if objs:
            counts = {}
            for o in objs:
                counts[o['label']] = counts.get(o['label'], 0) + 1
            summary = " | ".join([f"{k}: {v}" for k, v in counts.items()])
            cv2.putText(frame, summary, (px, py),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 170, 220), 1, cv2.LINE_AA)
        
        # Timestamp bar at bottom
        cv2.rectangle(frame, (0, h - 22), (w, h), (10, 10, 10), -1)
        cv2.putText(frame, f"{timestamp}  |  Session: {session_start}  |  Output: {csv_filename}",
                   (10, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
        
        # Update shared data
        with data_lock:
            frame_data['objects'] = obj_data_list
            frame_data['fps'] = round(avg_fps, 1)
            frame_data['frame_count'] = frame_num
            frame_data['frame_hsv'] = full_hsv
            frame_data['timestamp'] = timestamp
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def generate_hsv_view():
    """HSV channel decomposition view"""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        h_colored = cv2.applyColorMap((hsv[:,:,0] * 255 // 180).astype(np.uint8), cv2.COLORMAP_HSV)
        s_colored = cv2.applyColorMap(hsv[:,:,1], cv2.COLORMAP_VIRIDIS)
        v_gray = cv2.cvtColor(hsv[:,:,2], cv2.COLOR_GRAY2BGR)
        
        h_small = cv2.resize(h_colored, (320, 240))
        s_small = cv2.resize(s_colored, (320, 240))
        v_small = cv2.resize(v_gray, (320, 240))
        orig_small = cv2.resize(frame, (320, 240))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        for img, text in [(orig_small, "ORIGINAL"), (h_small, "HUE [0-180]"),
                          (s_small, "SATURATION [0-255]"), (v_small, "VALUE [0-255]")]:
            cv2.rectangle(img, (0, 0), (200, 28), (0, 0, 0), -1)
            cv2.putText(img, text, (8, 20), font, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        
        top = np.hstack([orig_small, h_small])
        bottom = np.hstack([s_small, v_small])
        grid = np.vstack([top, bottom])
        
        ret, buffer = cv2.imencode('.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ===== HTML DASHBOARD =====
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLOv5 Detection Dashboard</title>
    <meta charset="UTF-8">
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --border: #30363d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --accent-green: #58a835;
            --accent-blue: #58a6ff;
            --accent-orange: #d29922;
            --accent-red: #f85149;
            --accent-cyan: #56d4dd;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            font-size: 14px;
            line-height: 1.5;
        }
        
        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: 0.3px;
        }
        
        .header-status {
            display: flex;
            gap: 16px;
            align-items: center;
        }
        
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
            margin-right: 6px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        
        .header-stat {
            font-size: 12px;
            color: var(--text-secondary);
        }
        
        .header-stat strong {
            color: var(--accent-green);
            font-size: 14px;
        }
        
        .main-container {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 0;
            height: calc(100vh - 49px);
        }
        
        .video-section {
            padding: 16px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        
        .tab-bar {
            display: flex;
            gap: 2px;
            margin-bottom: 12px;
            background: var(--bg-secondary);
            border-radius: 6px;
            padding: 3px;
            width: fit-content;
        }
        
        .tab-btn {
            padding: 6px 18px;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            border-radius: 4px;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.15s;
        }
        
        .tab-btn:hover { color: var(--text-primary); background: var(--bg-tertiary); }
        .tab-btn.active { background: var(--bg-tertiary); color: var(--text-primary); }
        
        .stream-container {
            flex: 1;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .stream-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        
        .sidebar {
            background: var(--bg-secondary);
            border-left: 1px solid var(--border);
            overflow-y: auto;
            padding: 0;
        }
        
        .sidebar-section {
            border-bottom: 1px solid var(--border);
            padding: 14px 16px;
        }
        
        .sidebar-section:last-child { border-bottom: none; }
        
        .section-title {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-muted);
            margin-bottom: 10px;
        }
        
        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 3px 0;
            font-size: 13px;
        }
        
        .metric-label { color: var(--text-secondary); }
        .metric-value { color: var(--text-primary); font-weight: 500; font-family: 'SF Mono', monospace; font-size: 12px; }
        .metric-value.green { color: var(--accent-green); }
        .metric-value.blue { color: var(--accent-blue); }
        .metric-value.orange { color: var(--accent-orange); }
        .metric-value.cyan { color: var(--accent-cyan); }
        
        .hsv-row {
            display: grid;
            grid-template-columns: 24px 1fr 50px;
            gap: 8px;
            align-items: center;
            margin: 4px 0;
        }
        
        .hsv-label {
            font-size: 11px;
            font-weight: 600;
            font-family: 'SF Mono', monospace;
        }
        
        .hsv-bar-track {
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            overflow: hidden;
        }
        
        .hsv-bar-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s;
        }
        
        .hsv-val {
            font-size: 11px;
            font-family: 'SF Mono', monospace;
            color: var(--text-secondary);
            text-align: right;
        }
        
        .object-card {
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }
        
        .object-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .object-label {
            font-weight: 600;
            font-size: 13px;
            color: var(--text-primary);
        }
        
        .object-score {
            font-size: 11px;
            font-family: 'SF Mono', monospace;
            padding: 2px 8px;
            border-radius: 10px;
            font-weight: 600;
        }
        
        .score-high { background: rgba(88, 168, 53, 0.15); color: var(--accent-green); }
        .score-mid { background: rgba(210, 153, 34, 0.15); color: var(--accent-orange); }
        .score-low { background: rgba(248, 81, 73, 0.15); color: var(--accent-red); }
        
        .color-strip {
            display: flex;
            gap: 6px;
            margin: 6px 0;
            align-items: center;
        }
        
        .color-swatch {
            width: 14px;
            height: 14px;
            border-radius: 3px;
            border: 1px solid rgba(255,255,255,0.15);
            flex-shrink: 0;
        }
        
        .color-detail {
            font-size: 11px;
            color: var(--text-secondary);
            font-family: 'SF Mono', monospace;
        }
        
        .sub-section {
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid var(--border);
        }
        
        .sub-title {
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 4px;
        }
        
        .moment-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2px 12px;
            font-size: 11px;
            font-family: 'SF Mono', monospace;
        }
        
        .moment-grid .m-label { color: var(--text-muted); }
        .moment-grid .m-value { color: var(--accent-cyan); text-align: right; }
        
        .no-objects {
            text-align: center;
            padding: 30px 20px;
            color: var(--text-muted);
            font-size: 13px;
        }
        
        .export-btn {
            width: 100%;
            padding: 8px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            cursor: pointer;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.15s;
        }
        
        .export-btn:hover {
            background: var(--accent-green);
            color: #000;
            border-color: var(--accent-green);
        }
        
        .footer-bar {
            display: flex;
            justify-content: space-between;
            padding: 4px 12px;
            font-size: 11px;
            color: var(--text-muted);
            background: var(--bg-secondary);
            border-top: 1px solid var(--border);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>YOLOv5 Detection Dashboard</h1>
        <div class="header-status">
            <div class="header-stat"><span class="status-dot"></span>LIVE</div>
            <div class="header-stat">FPS: <strong id="h-fps">0</strong></div>
            <div class="header-stat">Objects: <strong id="h-count">0</strong></div>
            <div class="header-stat">Frame: <strong id="h-frame">0</strong></div>
        </div>
    </div>
    
    <div class="main-container">
        <div class="video-section">
            <div class="tab-bar">
                <button class="tab-btn active" onclick="switchView('detection', this)">Detection</button>
                <button class="tab-btn" onclick="switchView('hsv', this)">HSV Channels</button>
            </div>
            <div class="stream-container">
                <img id="stream" src="/video_feed">
            </div>
        </div>
        
        <div class="sidebar">
            <div class="sidebar-section">
                <div class="section-title">Frame HSV Statistics</div>
                <div id="frame-hsv">
                    <div class="hsv-row">
                        <span class="hsv-label" style="color: #e6a817">H</span>
                        <div class="hsv-bar-track"><div class="hsv-bar-fill" id="fh-bar" style="width:0%; background:#e6a817"></div></div>
                        <span class="hsv-val" id="fh-val">0</span>
                    </div>
                    <div class="hsv-row">
                        <span class="hsv-label" style="color: #40b040">S</span>
                        <div class="hsv-bar-track"><div class="hsv-bar-fill" id="fs-bar" style="width:0%; background:#40b040"></div></div>
                        <span class="hsv-val" id="fs-val">0</span>
                    </div>
                    <div class="hsv-row">
                        <span class="hsv-label" style="color: #d0d0d0">V</span>
                        <div class="hsv-bar-track"><div class="hsv-bar-fill" id="fv-bar" style="width:0%; background:#d0d0d0"></div></div>
                        <span class="hsv-val" id="fv-val">0</span>
                    </div>
                </div>
            </div>
            
            <div class="sidebar-section">
                <div class="section-title">Detected Objects</div>
                <div id="objects-panel">
                    <div class="no-objects">Waiting for detections...</div>
                </div>
            </div>
            
            <div class="sidebar-section">
                <div class="section-title">Data Export</div>
                <div class="metric-row">
                    <span class="metric-label">Records collected</span>
                    <span class="metric-value green" id="csv-count">0</span>
                </div>
                <div style="margin-top: 8px">
                    <button class="export-btn" onclick="triggerExport()">Export CSV Now</button>
                </div>
                <div style="margin-top:6px; font-size:11px; color:var(--text-muted)">
                    Auto-exports on shutdown (Ctrl+C)
                </div>
            </div>
        </div>
    </div>

    <script>
        function switchView(type, btn) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById('stream').src = (type === 'detection' ? '/video_feed?' : '/hsv_feed?') + Date.now();
        }
        
        function triggerExport() {
            fetch('/api/export').then(r => r.json()).then(d => {
                alert('Exported ' + d.records + ' records to ' + d.filename);
            });
        }
        
        function getScoreClass(score) {
            if (score >= 0.7) return 'score-high';
            if (score >= 0.4) return 'score-mid';
            return 'score-low';
        }
        
        function formatMoment(val) {
            if (Math.abs(val) > 10000) return val.toExponential(2);
            if (Math.abs(val) < 0.0001 && val !== 0) return val.toExponential(3);
            return val.toFixed(2);
        }
        
        function updateData() {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    // Header
                    document.getElementById('h-fps').textContent = data.fps;
                    document.getElementById('h-count').textContent = data.objects.length;
                    document.getElementById('h-frame').textContent = data.frame_count;
                    
                    // Frame HSV
                    if (data.frame_hsv) {
                        let fh = data.frame_hsv;
                        document.getElementById('fh-bar').style.width = (fh.h_mean / 180 * 100) + '%';
                        document.getElementById('fh-val').textContent = fh.h_mean.toFixed(1) + ' +/-' + fh.h_std.toFixed(1);
                        document.getElementById('fs-bar').style.width = (fh.s_mean / 255 * 100) + '%';
                        document.getElementById('fs-val').textContent = fh.s_mean.toFixed(1) + ' +/-' + fh.s_std.toFixed(1);
                        document.getElementById('fv-bar').style.width = (fh.v_mean / 255 * 100) + '%';
                        document.getElementById('fv-val').textContent = fh.v_mean.toFixed(1) + ' +/-' + fh.v_std.toFixed(1);
                    }
                    
                    // Objects
                    let html = '';
                    if (data.objects.length === 0) {
                        html = '<div class="no-objects">No objects detected</div>';
                    }
                    
                    data.objects.forEach((obj, i) => {
                        let sc = getScoreClass(obj.score);
                        html += '<div class="object-card">';
                        html += '<div class="object-header">';
                        html += '<span class="object-label">' + obj.label + '</span>';
                        html += '<span class="object-score ' + sc + '">' + (obj.score * 100).toFixed(1) + '%</span>';
                        html += '</div>';
                        
                        // BBox
                        html += '<div style="font-size:11px; color:var(--text-muted); font-family:monospace">';
                        html += 'bbox: [' + obj.bbox.join(', ') + '] | ';
                        html += (obj.bbox[2]-obj.bbox[0]) + 'x' + (obj.bbox[3]-obj.bbox[1]) + 'px';
                        html += '</div>';
                        
                        // Colors
                        if (obj.colors && obj.colors.length > 0) {
                            html += '<div class="sub-section"><div class="sub-title">Dominant Colors</div>';
                            obj.colors.forEach(c => {
                                html += '<div class="color-strip">';
                                html += '<div class="color-swatch" style="background:rgb(' + c.rgb[0] + ',' + c.rgb[1] + ',' + c.rgb[2] + ')"></div>';
                                html += '<span class="color-detail">' + c.name + ' ' + c.percentage + '% | HSV(' + c.hsv[0] + ',' + c.hsv[1] + ',' + c.hsv[2] + ')</span>';
                                html += '</div>';
                            });
                            html += '</div>';
                        }
                        
                        // HSV
                        if (obj.hsv) {
                            html += '<div class="sub-section"><div class="sub-title">HSV Analysis</div>';
                            html += '<div class="hsv-row"><span class="hsv-label" style="color:#e6a817">H</span>';
                            html += '<div class="hsv-bar-track"><div class="hsv-bar-fill" style="width:' + (obj.hsv.h_mean/180*100) + '%;background:#e6a817"></div></div>';
                            html += '<span class="hsv-val">' + obj.hsv.h_mean.toFixed(1) + '</span></div>';
                            html += '<div class="hsv-row"><span class="hsv-label" style="color:#40b040">S</span>';
                            html += '<div class="hsv-bar-track"><div class="hsv-bar-fill" style="width:' + (obj.hsv.s_mean/255*100) + '%;background:#40b040"></div></div>';
                            html += '<span class="hsv-val">' + obj.hsv.s_mean.toFixed(1) + '</span></div>';
                            html += '<div class="hsv-row"><span class="hsv-label" style="color:#d0d0d0">V</span>';
                            html += '<div class="hsv-bar-track"><div class="hsv-bar-fill" style="width:' + (obj.hsv.v_mean/255*100) + '%;background:#d0d0d0"></div></div>';
                            html += '<span class="hsv-val">' + obj.hsv.v_mean.toFixed(1) + '</span></div>';
                            html += '<div style="font-size:10px;color:var(--text-muted);font-family:monospace;margin-top:4px">';
                            html += 'Range H[' + obj.hsv.h_min + '-' + obj.hsv.h_max + '] ';
                            html += 'S[' + obj.hsv.s_min + '-' + obj.hsv.s_max + '] ';
                            html += 'V[' + obj.hsv.v_min + '-' + obj.hsv.v_max + ']';
                            html += '</div></div>';
                        }
                        
                        // Moments
                        if (obj.moments) {
                            let m = obj.moments;
                            html += '<div class="sub-section"><div class="sub-title">Spatial Moments</div>';
                            html += '<div class="moment-grid">';
                            html += '<span class="m-label">M00 (area)</span><span class="m-value">' + formatMoment(m.m00) + '</span>';
                            html += '<span class="m-label">M10</span><span class="m-value">' + formatMoment(m.m10) + '</span>';
                            html += '<span class="m-label">M01</span><span class="m-value">' + formatMoment(m.m01) + '</span>';
                            html += '<span class="m-label">M20</span><span class="m-value">' + formatMoment(m.m20) + '</span>';
                            html += '<span class="m-label">M11</span><span class="m-value">' + formatMoment(m.m11) + '</span>';
                            html += '<span class="m-label">M02</span><span class="m-value">' + formatMoment(m.m02) + '</span>';
                            html += '</div>';
                            
                            html += '<div class="sub-title" style="margin-top:6px">Central Moments (2nd Order)</div>';
                            html += '<div class="moment-grid">';
                            html += '<span class="m-label">mu20</span><span class="m-value">' + formatMoment(m.mu20) + '</span>';
                            html += '<span class="m-label">mu11</span><span class="m-value">' + formatMoment(m.mu11) + '</span>';
                            html += '<span class="m-label">mu02</span><span class="m-value">' + formatMoment(m.mu02) + '</span>';
                            html += '</div>';
                            
                            html += '<div class="sub-title" style="margin-top:6px">Normalized Central Moments</div>';
                            html += '<div class="moment-grid">';
                            html += '<span class="m-label">nu20</span><span class="m-value">' + m.nu20.toExponential(4) + '</span>';
                            html += '<span class="m-label">nu11</span><span class="m-value">' + m.nu11.toExponential(4) + '</span>';
                            html += '<span class="m-label">nu02</span><span class="m-value">' + m.nu02.toExponential(4) + '</span>';
                            html += '</div>';
                            
                            html += '<div class="sub-title" style="margin-top:6px">Derived</div>';
                            html += '<div class="moment-grid">';
                            html += '<span class="m-label">Centroid X</span><span class="m-value">' + m.centroid_x.toFixed(2) + '</span>';
                            html += '<span class="m-label">Centroid Y</span><span class="m-value">' + m.centroid_y.toFixed(2) + '</span>';
                            html += '<span class="m-label">Orientation</span><span class="m-value">' + m.orientation.toFixed(2) + ' deg</span>';
                            html += '</div>';
                            
                            if (m.hu_moments) {
                                html += '<div class="sub-title" style="margin-top:6px">Hu Moments (log)</div>';
                                html += '<div class="moment-grid">';
                                m.hu_moments.forEach((h, idx) => {
                                    html += '<span class="m-label">Hu[' + idx + ']</span><span class="m-value">' + h.toFixed(4) + '</span>';
                                });
                                html += '</div>';
                            }
                            html += '</div>';
                        }
                        
                        html += '</div>';
                    });
                    
                    document.getElementById('objects-panel').innerHTML = html;
                })
                .catch(err => {});
            
            fetch('/api/csv_count').then(r => r.json()).then(d => {
                document.getElementById('csv-count').textContent = d.count;
            }).catch(err => {});
        }
        
        setInterval(updateData, 600);
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/hsv_feed')
def hsv_feed():
    return Response(generate_hsv_view(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/data')
def api_data():
    with data_lock:
        return json.dumps(frame_data, default=str)


@app.route('/api/csv_count')
def api_csv_count():
    with csv_lock:
        return json.dumps({'count': len(csv_records)})


@app.route('/api/export')
def api_export():
    export_csv()
    with csv_lock:
        return json.dumps({'status': 'ok', 'records': len(csv_records), 'filename': csv_filename})


@app.route('/download_csv')
def download_csv():
    export_csv()
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return "No data yet", 404


if __name__ == '__main__':
    print("=" * 60)
    print("  YOLOv5 Detection Dashboard")
    print("  Session: " + session_start)
    print("  CSV Output: " + csv_filename)
    print("  Open browser: http://<jetson-ip>:5000")
    print("  Press Ctrl+C to stop and export CSV")
    print("=" * 60)
    
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[INFO] Shutting down. Exporting CSV...")
        export_csv()
        cap.release()
        print("[INFO] Done.")
ENDOFSCRIPT