from flask import Flask, Response, render_template_string
import cv2
import numpy as np
from elements.yolo import OBJ_DETECTION
from collections import Counter
import json
import threading
import time

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

# Shared data between threads
frame_data = {
    'objects': [],
    'fps': 0,
    'frame_count': 0
}
data_lock = threading.Lock()


def get_dominant_colors(image, k=3):
    """Get dominant colors from an image region using k-means"""
    if image.size == 0 or image.shape[0] < 2 or image.shape[1] < 2:
        return []
    
    # Resize for speed
    small = cv2.resize(image, (30, 30))
    pixels = small.reshape(-1, 3).astype(np.float32)
    
    # K-means clustering
    k = min(k, len(pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    try:
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        
        # Count pixels per cluster
        label_counts = Counter(labels.flatten())
        total = sum(label_counts.values())
        
        colors = []
        for idx, count in label_counts.most_common(k):
            bgr = centers[idx].astype(int)
            rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
            hsv_pixel = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            percentage = round(count / total * 100, 1)
            
            # Get color name from HSV
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
    """Convert HSV values to human-readable color name"""
    if v < 40:
        return "Black"
    if s < 30:
        if v > 200:
            return "White"
        elif v > 120:
            return "Light Gray"
        else:
            return "Dark Gray"
    
    # Classify by hue
    if h < 10 or h > 170:
        return "Red"
    elif h < 22:
        return "Orange"
    elif h < 35:
        return "Yellow"
    elif h < 78:
        return "Green"
    elif h < 100:
        return "Cyan"
    elif h < 130:
        return "Blue"
    elif h < 145:
        return "Purple"
    elif h < 170:
        return "Pink"
    return "Unknown"


def calculate_moments(mask_or_contour, roi):
    """Calculate image moments for a region"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(gray)
    
    result = {
        'area': round(moments['m00'], 1),
        'centroid_x': 0,
        'centroid_y': 0,
        'orientation': 0,
        'hu_moments': []
    }
    
    if moments['m00'] != 0:
        result['centroid_x'] = round(moments['m10'] / moments['m00'], 1)
        result['centroid_y'] = round(moments['m01'] / moments['m00'], 1)
    
    # Calculate orientation
    if moments['mu20'] - moments['mu02'] != 0:
        result['orientation'] = round(
            0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02']) * 180 / np.pi, 1
        )
    
    # Hu moments (log scale)
    hu = cv2.HuMoments(moments)
    result['hu_moments'] = [round(float(-np.sign(h) * np.log10(abs(h) + 1e-10)), 2) for h in hu.flatten()]
    
    return result


def get_hsv_stats(roi):
    """Get HSV statistics for a region"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    return {
        'h_mean': round(float(np.mean(hsv[:,:,0])), 1),
        'h_std': round(float(np.std(hsv[:,:,0])), 1),
        's_mean': round(float(np.mean(hsv[:,:,1])), 1),
        's_std': round(float(np.std(hsv[:,:,1])), 1),
        'v_mean': round(float(np.mean(hsv[:,:,2])), 1),
        'v_std': round(float(np.std(hsv[:,:,2])), 1),
        'h_min': int(np.min(hsv[:,:,0])),
        'h_max': int(np.max(hsv[:,:,0])),
        's_min': int(np.min(hsv[:,:,1])),
        's_max': int(np.max(hsv[:,:,1])),
        'v_min': int(np.min(hsv[:,:,2])),
        'v_max': int(np.max(hsv[:,:,2]))
    }


def draw_color_bar(frame, colors, x, y, bar_width=80, bar_height=15):
    """Draw color bar on frame"""
    current_x = x
    for color_info in colors:
        r, g, b = color_info['rgb']
        width = int(bar_width * color_info['percentage'] / 100)
        if width < 1:
            width = 1
        cv2.rectangle(frame, (current_x, y), (current_x + width, y + bar_height),
                     (b, g, r), -1)
        current_x += width
    # Border
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 1)


def draw_hsv_bar(frame, hsv_stats, x, y):
    """Draw HSV visualization bar"""
    bar_width = 100
    bar_height = 8
    
    # H bar (0-180)
    h_fill = int(bar_width * hsv_stats['h_mean'] / 180)
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + h_fill, y + bar_height), (0, 255, 255), -1)
    cv2.putText(frame, f"H:{int(hsv_stats['h_mean'])}", (x + bar_width + 5, y + bar_height),
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    
    # S bar (0-255)
    y += bar_height + 3
    s_fill = int(bar_width * hsv_stats['s_mean'] / 255)
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + s_fill, y + bar_height), (0, 255, 0), -1)
    cv2.putText(frame, f"S:{int(hsv_stats['s_mean'])}", (x + bar_width + 5, y + bar_height),
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    # V bar (0-255)
    y += bar_height + 3
    v_fill = int(bar_width * hsv_stats['v_mean'] / 255)
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + v_fill, y + bar_height), (255, 255, 255), -1)
    cv2.putText(frame, f"V:{int(hsv_stats['v_mean'])}", (x + bar_width + 5, y + bar_height),
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


def draw_moment_info(frame, moments_data, cx, cy):
    """Draw moment centroid marker"""
    # Draw crosshair at centroid
    size = 8
    cv2.line(frame, (cx - size, cy), (cx + size, cy), (0, 255, 255), 1)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), (0, 255, 255), 1)
    cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)


def generate_frames():
    """Generate frames with full analysis overlay"""
    prev_time = time.time()
    fps_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time + 0.001)
        prev_time = current_time
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)
        
        # YOLO Detection
        objs = Object_detector.detect(frame)
        
        # Full frame HSV stats
        full_hsv = get_hsv_stats(frame)
        
        obj_data_list = []
        
        for obj in objs:
            label = obj['label']
            score = obj['score']
            [(xmin, ymin), (xmax, ymax)] = obj['bbox']
            color = Object_colors[Object_classes.index(label)]
            
            # Ensure bounds are valid
            h, w = frame.shape[:2]
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            
            # Extract ROI
            roi = frame[ymin:ymax, xmin:xmax]
            
            if roi.size > 0:
                # Get analysis data
                dominant_colors = get_dominant_colors(roi, k=3)
                hsv_stats = get_hsv_stats(roi)
                moments_data = calculate_moments(None, roi)
                
                # Store for API
                obj_data_list.append({
                    'label': label,
                    'score': score,
                    'bbox': [xmin, ymin, xmax, ymax],
                    'colors': dominant_colors,
                    'hsv': hsv_stats,
                    'moments': moments_data
                })
                
                # ===== DRAW BOUNDING BOX =====
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                
                # ===== DRAW LABEL WITH SCORE =====
                label_text = f'{label} {score:.0%}'
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, (xmin, ymin - 20), (xmin + label_size[0] + 5, ymin), color, -1)
                cv2.putText(frame, label_text, (xmin + 2, ymin - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # ===== DRAW DOMINANT COLORS =====
                if dominant_colors:
                    # Color bar below bounding box
                    draw_color_bar(frame, dominant_colors, xmin, ymax + 2)
                    
                    # Color names
                    color_text = ", ".join([f"{c['name']}({c['percentage']}%)" for c in dominant_colors[:2]])
                    cv2.putText(frame, color_text, (xmin, ymax + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                
                # ===== DRAW HSV BARS =====
                draw_hsv_bar(frame, hsv_stats, xmax + 5, ymin)
                
                # ===== DRAW MOMENT CENTROID =====
                if moments_data['area'] > 0:
                    abs_cx = xmin + int(moments_data['centroid_x'])
                    abs_cy = ymin + int(moments_data['centroid_y'])
                    draw_moment_info(frame, moments_data, abs_cx, abs_cy)
                    
                    # Moment text
                    cv2.putText(frame, f"Area:{int(moments_data['area'])}", (xmin, ymin - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                    cv2.putText(frame, f"Angle:{moments_data['orientation']}deg", (xmin, ymin - 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        
        # ===== DRAW GLOBAL INFO PANEL =====
        panel_x = 10
        panel_y = 10
        panel_w = 220
        panel_h = 140
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # FPS
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (panel_x + 10, panel_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Object count
        cv2.putText(frame, f"Objects: {len(objs)}", (panel_x + 10, panel_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Full frame HSV
        cv2.putText(frame, f"Frame HSV:", (panel_x + 10, panel_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"  H: {full_hsv['h_mean']:.0f} (std:{full_hsv['h_std']:.0f})", (panel_x + 10, panel_y + 78),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"  S: {full_hsv['s_mean']:.0f} (std:{full_hsv['s_std']:.0f})", (panel_x + 10, panel_y + 93),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"  V: {full_hsv['v_mean']:.0f} (std:{full_hsv['v_std']:.0f})", (panel_x + 10, panel_y + 108),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Detected classes summary
        if objs:
            class_counts = {}
            for obj in objs:
                l = obj['label']
                class_counts[l] = class_counts.get(l, 0) + 1
            summary = ", ".join([f"{v}x{k}" for k, v in class_counts.items()])
            cv2.putText(frame, f"{summary}", (panel_x + 10, panel_y + 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1, cv2.LINE_AA)
        
        # Update shared data
        with data_lock:
            frame_data['objects'] = obj_data_list
            frame_data['fps'] = round(avg_fps, 1)
            frame_data['frame_count'] += 1
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def generate_hsv_view():
    """Generate HSV visualization stream"""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create 3 channel views
        h_channel = hsv[:,:,0]
        s_channel = hsv[:,:,1]
        v_channel = hsv[:,:,2]
        
        # Colorize each channel
        h_colored = cv2.applyColorMap((h_channel * 255 // 180).astype(np.uint8), cv2.COLORMAP_HSV)
        s_colored = cv2.applyColorMap(s_channel, cv2.COLORMAP_VIRIDIS)
        v_colored = cv2.cvtColor(v_channel, cv2.COLOR_GRAY2BGR)
        
        # Resize for grid
        h_small = cv2.resize(h_colored, (320, 240))
        s_small = cv2.resize(s_colored, (320, 240))
        v_small = cv2.resize(v_colored, (320, 240))
        orig_small = cv2.resize(frame, (320, 240))
        
        # Add labels
        cv2.putText(h_small, "HUE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(s_small, "SATURATION", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(v_small, "VALUE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(orig_small, "ORIGINAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Stack into grid
        top = np.hstack([orig_small, h_small])
        bottom = np.hstack([s_small, v_small])
        grid = np.vstack([top, bottom])
        
        ret, buffer = cv2.imencode('.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ===== HTML DASHBOARD =====
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLOv5 Jetson Nano Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #1a1a2e; color: #eee; font-family: 'Segoe UI', Arial, sans-serif; }
        .header { background: #16213e; padding: 15px; text-align: center; border-bottom: 2px solid #0f3460; }
        .header h1 { color: #76b900; font-size: 24px; }
        .container { display: flex; flex-wrap: wrap; padding: 10px; gap: 10px; }
        .video-panel { flex: 2; min-width: 640px; }
        .info-panel { flex: 1; min-width: 300px; }
        .stream { width: 100%; border: 2px solid #0f3460; border-radius: 8px; }
        .card { background: #16213e; border-radius: 8px; padding: 15px; margin-bottom: 10px; border: 1px solid #0f3460; }
        .card h3 { color: #76b900; margin-bottom: 10px; border-bottom: 1px solid #0f3460; padding-bottom: 5px; }
        .stat { display: flex; justify-content: space-between; padding: 4px 0; }
        .stat-label { color: #a0a0a0; }
        .stat-value { color: #76b900; font-weight: bold; }
        .color-box { display: inline-block; width: 20px; height: 20px; border-radius: 3px; margin-right: 5px; vertical-align: middle; border: 1px solid #fff; }
        .obj-item { background: #1a1a2e; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 3px solid #76b900; }
        .hsv-bar { height: 12px; border-radius: 3px; margin: 3px 0; }
        .tabs { display: flex; gap: 5px; margin-bottom: 10px; }
        .tab { padding: 8px 16px; background: #0f3460; border: none; color: #eee; cursor: pointer; border-radius: 5px; }
        .tab.active { background: #76b900; color: #000; }
        .tab:hover { background: #76b900; color: #000; }
        #data-panel { max-height: 500px; overflow-y: auto; }
        .moment-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 YOLOv5 Jetson Nano Detection Dashboard</h1>
    </div>
    <div class="container">
        <div class="video-panel">
            <div class="tabs">
                <button class="tab active" onclick="switchStream('detection')">Detection</button>
                <button class="tab" onclick="switchStream('hsv')">HSV View</button>
            </div>
            <img id="stream" class="stream" src="/video_feed">
        </div>
        <div class="info-panel">
            <div class="card">
                <h3>📊 System Stats</h3>
                <div class="stat"><span class="stat-label">FPS:</span><span class="stat-value" id="fps">0</span></div>
                <div class="stat"><span class="stat-label">Objects:</span><span class="stat-value" id="obj-count">0</span></div>
                <div class="stat"><span class="stat-label">Frame:</span><span class="stat-value" id="frame-count">0</span></div>
            </div>
            <div class="card">
                <h3>🎯 Detected Objects</h3>
                <div id="data-panel">
                    <p style="color: #a0a0a0;">Waiting for data...</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        function switchStream(type) {
            const img = document.getElementById('stream');
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            
            if (type === 'detection') {
                img.src = '/video_feed?' + Date.now();
            } else {
                img.src = '/hsv_feed?' + Date.now();
            }
        }

        function updateData() {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps;
                    document.getElementById('obj-count').textContent = data.objects.length;
                    document.getElementById('frame-count').textContent = data.frame_count;
                    
                    let html = '';
                    if (data.objects.length === 0) {
                        html = '<p style="color: #a0a0a0;">No objects detected</p>';
                    }
                    
                    data.objects.forEach((obj, i) => {
                        html += '<div class="obj-item">';
                        html += '<strong>' + obj.label + '</strong> (' + (obj.score * 100).toFixed(0) + '%)';
                        
                        // Colors
                        if (obj.colors && obj.colors.length > 0) {
                            html += '<div style="margin-top:5px"><small style="color:#a0a0a0">Dominant Colors:</small><br>';
                            obj.colors.forEach(c => {
                                html += '<span class="color-box" style="background:rgb(' + c.rgb[0] + ',' + c.rgb[1] + ',' + c.rgb[2] + ')"></span>';
                                html += '<small>' + c.name + ' ' + c.percentage + '%</small> ';
                            });
                            html += '</div>';
                        }
                        
                        // HSV
                        if (obj.hsv) {
                            html += '<div style="margin-top:5px"><small style="color:#a0a0a0">HSV Values:</small>';
                            html += '<div class="hsv-bar" style="background:linear-gradient(to right, #000, hsl(' + (obj.hsv.h_mean * 2) + ',100%,50%)); width:' + (obj.hsv.h_mean / 180 * 100) + '%"></div>';
                            html += '<small>H:' + obj.hsv.h_mean + ' S:' + obj.hsv.s_mean + ' V:' + obj.hsv.v_mean + '</small>';
                            html += '</div>';
                        }
                        
                        // Moments
                        if (obj.moments) {
                            html += '<div style="margin-top:5px"><small style="color:#a0a0a0">Moments:</small>';
                            html += '<div class="moment-grid">';
                            html += '<small>Area: ' + Math.round(obj.moments.area) + '</small>';
                            html += '<small>Angle: ' + obj.moments.orientation + '°</small>';
                            html += '</div></div>';
                        }
                        
                        html += '</div>';
                    });
                    
                    document.getElementById('data-panel').innerHTML = html;
                })
                .catch(err => console.log('Error:', err));
        }
        
        setInterval(updateData, 500);
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


if __name__ == '__main__':
    print("Starting YOLOv5 Dashboard...")
    print("Open browser: http://<jetson-ip>:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
