import matplotlib
matplotlib.use('Agg')

from flask import Flask, Response, render_template_string
import cv2
import numpy as np
from elements.yolo import OBJ_DETECTION
import json
import threading
import time
from datetime import datetime
from collections import deque

app = Flask(__name__)

ALL_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush']

Object_detector = OBJ_DETECTION('weights/yolov5s.pt', ALL_CLASSES)


# ============================================================
# THREADED CAMERA
# ============================================================
class Cam:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = threading.Lock()
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        while True:
            ret, f = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = f

    def get(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.cap.release()

cam = Cam()

# Shared between threads
latest_jpg = None
jpg_lock = threading.Lock()
frame_data = {'persons': [], 'fps': 0, 'frame_count': 0}
data_lock = threading.Lock()

C_STAND = (53, 197, 34)
C_SIT = (0, 158, 245)
C_GRAY = (128, 128, 128)
C_SKEL = (220, 200, 0)


# ============================================================
# FAST CLASSIFIER - aspect ratio + relative height only
# ============================================================
def classify_posture(bbox, frame_h, frame_w):
    xmin, ymin, xmax, ymax = bbox
    bw = xmax - xmin
    bh = ymax - ymin
    if bw < 5 or bh < 5:
        return 'UNCERTAIN', 0.5, 0.5, {}

    ar = bh / bw
    rh = bh / frame_h
    bot = ymax / frame_h

    score = 0.5

    # Aspect ratio (strongest)
    if ar >= 2.5: score += 0.25
    elif ar >= 2.0: score += 0.15
    elif ar >= 1.7: score += 0.05
    elif ar <= 1.0: score -= 0.25
    elif ar <= 1.3: score -= 0.15
    elif ar <= 1.5: score -= 0.05

    # Relative height
    if rh > 0.7: score += 0.08
    elif rh > 0.5: score += 0.03
    elif rh < 0.25: score -= 0.06
    elif rh < 0.35: score -= 0.03

    score = max(0.0, min(1.0, score))

    features = {'aspect_ratio': round(ar, 3), 'rel_height': round(rh, 3), 'bottom': round(bot, 3)}

    if score >= 0.55:
        return 'STANDING', round(min(score, 0.99), 2), round(score, 3), features
    elif score <= 0.45:
        return 'SITTING', round(min(1 - score, 0.99), 2), round(score, 3), features
    else:
        return 'UNCERTAIN', 0.5, round(score, 3), features


# ============================================================
# SIMPLE TRACKER
# ============================================================
class Tracker:
    def __init__(self):
        self.persons = {}
        self.nid = 1

    def iou(self, a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        i = max(0, x2 - x1) * max(0, y2 - y1)
        u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
        return i / max(u, 1)

    def update(self, dets):
        now = time.time()
        matched_p = set()
        matched_d = set()

        # Match existing
        for pid in list(self.persons.keys()):
            best_iou = 0
            best_di = -1
            for di, d in enumerate(dets):
                if di in matched_d:
                    continue
                v = self.iou(self.persons[pid]['bbox'], d['bbox'])
                if v > best_iou:
                    best_iou = v
                    best_di = di
            if best_iou > 0.2 and best_di >= 0:
                p = self.persons[pid]
                d = dets[best_di]
                p['bbox'] = d['bbox']
                p['posture_raw'] = d['posture']
                p['confidence'] = d['confidence']
                p['raw_score'] = d['raw_score']
                p['features'] = d['features']
                p['hist'].append(d['posture'])
                p['gone'] = 0
                p['last'] = now

                # Smooth
                h = list(p['hist'])
                st = h.count('STANDING')
                si = h.count('SITTING')
                new_p = 'STANDING' if st > si else ('SITTING' if si > st else p['posture'])

                elapsed = now - p['p_start']
                if new_p != p['posture']:
                    if p['posture'] == 'STANDING': p['t_stand'] += elapsed
                    elif p['posture'] == 'SITTING': p['t_sit'] += elapsed
                    p['p_start'] = now
                    p['trans'] += 1
                    p['p_dur'] = 0
                else:
                    p['p_dur'] = elapsed
                p['posture'] = new_p

                matched_p.add(pid)
                matched_d.add(best_di)

        # New persons
        for di, d in enumerate(dets):
            if di not in matched_d:
                pid = self.nid; self.nid += 1
                self.persons[pid] = {
                    'id': pid, 'bbox': d['bbox'],
                    'posture_raw': d['posture'], 'posture': d['posture'],
                    'confidence': d['confidence'], 'raw_score': d['raw_score'],
                    'features': d['features'],
                    'hist': deque(maxlen=8),
                    'p_start': now, 't_stand': 0, 't_sit': 0,
                    'trans': 0, 'p_dur': 0, 'gone': 0, 'last': now
                }
                self.persons[pid]['hist'].append(d['posture'])

        # Remove old
        for pid in list(self.persons.keys()):
            if pid not in matched_p:
                self.persons[pid]['gone'] += 1
                if self.persons[pid]['gone'] > 15:
                    del self.persons[pid]

        return [v for v in self.persons.values() if v['gone'] == 0]


tracker = Tracker()

def get_color(p):
    if p == 'STANDING': return C_STAND
    if p == 'SITTING': return C_SIT
    return C_GRAY

def draw_skeleton(frame, bbox, posture):
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin; h = ymax - ymin
    cx = (xmin + xmax) // 2

    head = (cx, ymin + int(h * 0.08))
    neck = (cx, ymin + int(h * 0.16))
    ls = (cx - int(w * 0.24), ymin + int(h * 0.20))
    rs = (cx + int(w * 0.24), ymin + int(h * 0.20))

    if posture == 'STANDING':
        lh = (cx - int(w * 0.13), ymin + int(h * 0.50))
        rh = (cx + int(w * 0.13), ymin + int(h * 0.50))
        lk = (cx - int(w * 0.12), ymin + int(h * 0.74))
        rk = (cx + int(w * 0.12), ymin + int(h * 0.74))
        la = (cx - int(w * 0.12), ymin + int(h * 0.95))
        ra = (cx + int(w * 0.12), ymin + int(h * 0.95))
    else:
        lh = (cx - int(w * 0.15), ymin + int(h * 0.52))
        rh = (cx + int(w * 0.15), ymin + int(h * 0.52))
        lk = (cx - int(w * 0.24), ymin + int(h * 0.78))
        rk = (cx + int(w * 0.24), ymin + int(h * 0.78))
        la = (cx - int(w * 0.18), ymin + int(h * 0.95))
        ra = (cx + int(w * 0.18), ymin + int(h * 0.95))

    for a, b in [(head,neck),(neck,ls),(neck,rs),(ls,lh),(rs,rh),(lh,rh),(lh,lk),(rh,rk),(lk,la),(rk,ra)]:
        cv2.line(frame, a, b, C_SKEL, 1, cv2.LINE_AA)
    cv2.circle(frame, head, max(int(w * 0.08), 3), C_SKEL, -1)
    for pt in [neck, ls, rs, lh, rh, lk, rk, la, ra]:
        cv2.circle(frame, pt, 2, C_SKEL, -1)

def fmt(s):
    if s < 60: return "{}s".format(int(s))
    return "{}m{}s".format(int(s // 60), int(s % 60))


# ============================================================
# DETECTION THREAD - runs separately from streaming
# ============================================================
DETECT_EVERY = 5
last_results = []
results_lock = threading.Lock()
start_time = time.time()

def detection_loop():
    global last_results
    fnum = 0
    fps_buf = []
    prev = time.time()

    while True:
        frame = cam.get()
        if frame is None:
            time.sleep(0.05)
            continue

        fnum += 1
        now = time.time()
        fps = 1.0 / (now - prev + 0.0001)
        prev = now
        fps_buf.append(fps)
        if len(fps_buf) > 15: fps_buf.pop(0)
        avg_fps = sum(fps_buf) / len(fps_buf)

        fh, fw = frame.shape[:2]

        # Detect only every N frames
        if fnum % DETECT_EVERY == 0:
            objs = Object_detector.detect(frame)
            persons = [o for o in objs if o['label'] == 'person']

            dets = []
            for obj in persons:
                [(xmin, ymin), (xmax, ymax)] = obj['bbox']
                bbox = [xmin, ymin, xmax, ymax]
                posture, conf, raw, features = classify_posture(bbox, fh, fw)
                dets.append({'bbox': bbox, 'posture': posture, 'confidence': conf,
                            'raw_score': raw, 'features': features})

            tracked = tracker.update(dets)

            with results_lock:
                last_results = tracked

        # Draw on frame
        with results_lock:
            persons_to_draw = last_results

        t_stand = 0; t_sit = 0

        for p in persons_to_draw:
            bbox = p['bbox']
            xmin, ymin, xmax, ymax = bbox
            posture = p['posture']
            color = get_color(posture)

            ps = p.get('t_stand', 0)
            pi = p.get('t_sit', 0)
            if posture == 'STANDING': ps += p.get('p_dur', 0)
            elif posture == 'SITTING': pi += p.get('p_dur', 0)
            t_stand += ps; t_sit += pi

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            # Corners
            cl = min(12, (xmax - xmin) // 5)
            for cx, cy, dx, dy in [(xmin,ymin,1,1),(xmax,ymin,-1,1),(xmin,ymax,1,-1),(xmax,ymax,-1,-1)]:
                cv2.line(frame, (cx, cy), (cx + cl * dx, cy), color, 2)
                cv2.line(frame, (cx, cy), (cx, cy + cl * dy), color, 2)

            # Label
            lbl = "#{} {} {:.0f}%".format(p['id'], posture, p['confidence'] * 100)
            sz = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
            cv2.rectangle(frame, (xmin, ymin - 16), (xmin + sz[0] + 4, ymin - 1), color, -1)
            cv2.putText(frame, lbl, (xmin + 2, ymin - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

            # Duration
            cv2.putText(frame, fmt(p.get('p_dur', 0)), (xmin, ymax + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

            # AR value
            ar = p.get('features', {}).get('aspect_ratio', 0)
            cv2.putText(frame, "AR:{:.1f}".format(ar), (xmin, ymax + 24),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1, cv2.LINE_AA)

            draw_skeleton(frame, bbox, posture)

        # Info panel
        elapsed = now - start_time
        ov = frame.copy()
        cv2.rectangle(ov, (3, 3), (200, 75), (15, 15, 15), -1)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "POSTURE DETECTION", (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, "FPS:{:.1f} F:{}".format(avg_fps, fnum), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 180, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "Persons:{} T:{}".format(len(persons_to_draw), fmt(elapsed)), (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, "Stand:{}".format(fmt(t_stand)), (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_STAND, 1, cv2.LINE_AA)
        cv2.putText(frame, "Sit:{}".format(fmt(t_sit)), (100, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_SIT, 1, cv2.LINE_AA)

        # Ratio bar
        total_t = t_stand + t_sit
        if total_t > 0:
            sw = int(180 * t_stand / total_t)
            cv2.rectangle(frame, (8, 56), (8 + sw, 61), C_STAND, -1)
            cv2.rectangle(frame, (8 + sw, 56), (188, 61), C_SIT, -1)
        cv2.rectangle(frame, (8, 56), (188, 61), (60, 60, 60), 1)

        # Bottom bar
        cv2.rectangle(frame, (0, fh - 14), (fw, fh), (15, 15, 15), -1)
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (4, fh - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1, cv2.LINE_AA)

        # Update shared data
        pdata = []
        for p in persons_to_draw:
            pdata.append({
                'id': p['id'], 'posture': p['posture'],
                'confidence': p['confidence'], 'raw_score': p.get('raw_score', 0.5),
                'features': p.get('features', {}),
                'duration': round(p.get('p_dur', 0), 1),
                'total_standing': round(p.get('t_stand', 0) + (p.get('p_dur', 0) if p['posture'] == 'STANDING' else 0), 1),
                'total_sitting': round(p.get('t_sit', 0) + (p.get('p_dur', 0) if p['posture'] == 'SITTING' else 0), 1),
                'transitions': p.get('trans', 0)
            })

        with data_lock:
            frame_data['persons'] = pdata
            frame_data['fps'] = round(avg_fps, 1)
            frame_data['frame_count'] = fnum

        # Encode
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if ret:
            with jpg_lock:
                global latest_jpg
                latest_jpg = buf.tobytes()

        # Small sleep to not hog CPU
        time.sleep(0.01)


# Start detection in background thread
det_thread = threading.Thread(target=detection_loop, daemon=True)
det_thread.start()


def generate_frames():
    while True:
        with jpg_lock:
            if latest_jpg is None:
                time.sleep(0.05)
                continue
            jpg = latest_jpg
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        time.sleep(0.03)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Posture Detection</title>
    <style>
        :root{--bg:#0b0e14;--bg1:#111620;--bg2:#1a1f2e;--brd:#2d3548;--t0:#e0e4ec;--t1:#a0a8b8;--t2:#6c7589;--grn:#34d058;--gd:rgba(52,208,88,.12);--org:#f5a623;--od:rgba(245,166,35,.12);--red:#e53e3e;--cyan:#4dc9f6}
        *{margin:0;padding:0;box-sizing:border-box}
        body{background:var(--bg);color:var(--t0);font-family:-apple-system,sans-serif;font-size:13px}
        .h{background:var(--bg1);border-bottom:1px solid var(--brd);padding:8px 16px;display:flex;justify-content:space-between;align-items:center}
        .h h1{font-size:14px;font-weight:600}
        .hr{display:flex;gap:14px;font-size:11px;color:var(--t1)}
        .hr b{color:var(--grn);font-family:monospace}
        .dot{width:6px;height:6px;border-radius:50%;background:var(--grn);display:inline-block;margin-right:4px;animation:b 1.5s infinite}
        @keyframes b{0%,100%{opacity:1}50%{opacity:.3}}
        .l{display:grid;grid-template-columns:1fr 300px;height:calc(100vh - 38px)}
        .v{padding:8px;display:flex}
        .sw{flex:1;background:#000;border-radius:5px;overflow:hidden;border:1px solid var(--brd)}
        .sw img{width:100%;height:100%;object-fit:contain}
        .sb{background:var(--bg1);border-left:1px solid var(--brd);overflow-y:auto;padding:10px}
        .sec{margin-bottom:10px}
        .st{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--t2);margin-bottom:6px}
        .sr{display:flex;justify-content:space-between;padding:2px 0;font-size:11px}
        .sl{color:var(--t1)}.sv{font-family:monospace;font-size:10px}
        .rb{height:6px;border-radius:3px;overflow:hidden;background:var(--bg2);margin:4px 0;display:flex}
        .rg{background:var(--grn);transition:width .5s}.ro{background:var(--org);transition:width .5s}
        .pc{background:var(--bg2);border:1px solid var(--brd);border-radius:5px;padding:8px;margin-bottom:6px}
        .ph{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
        .pid{font-weight:600;font-size:12px}
        .bd{font-size:9px;font-weight:700;padding:2px 8px;border-radius:8px}
        .bs{background:var(--gd);color:var(--grn)}.bi{background:var(--od);color:var(--org)}.bu{background:rgba(108,117,137,.2);color:var(--t2)}
        .pm{font-size:10px;color:var(--t2);font-family:monospace}
        .scb{height:6px;border-radius:3px;background:var(--bg2);margin:4px 0;overflow:hidden}
        .scf{height:100%;border-radius:3px;transition:width .3s}
        .nd{text-align:center;padding:16px;color:var(--t2);font-size:11px}
        .ab{background:rgba(229,62,62,.1);border:1px solid rgba(229,62,62,.2);border-radius:4px;padding:4px 8px;margin-bottom:4px;font-size:10px;color:var(--red)}
    </style>
</head>
<body>
    <div class="h">
        <h1>POSTURE DETECTION</h1>
        <div class="hr">
            <span><span class="dot"></span>LIVE</span>
            <span>FPS: <b id="hf">0</b></span>
            <span>Persons: <b id="hc">0</b></span>
        </div>
    </div>
    <div class="l">
        <div class="v"><div class="sw"><img id="s" src="/video_feed"></div></div>
        <div class="sb">
            <div class="sec">
                <div class="st">Session</div>
                <div class="sr"><span class="sl">Standing</span><span class="sv" style="color:var(--grn)" id="ts">0s</span></div>
                <div class="sr"><span class="sl">Sitting</span><span class="sv" style="color:var(--org)" id="ti">0s</span></div>
                <div class="rb"><div class="rg" id="rg" style="width:50%"></div><div class="ro" id="ro" style="width:50%"></div></div>
            </div>
            <div id="al"></div>
            <div class="sec">
                <div class="st">Persons</div>
                <div id="pp"><div class="nd">Waiting...</div></div>
            </div>
        </div>
    </div>
    <script>
        function f(s){if(s<60)return Math.round(s)+'s';return Math.floor(s/60)+'m'+Math.round(s%60)+'s'}
        function u(){
            fetch('/api/data').then(r=>r.json()).then(d=>{
                document.getElementById('hf').textContent=d.fps;
                document.getElementById('hc').textContent=d.persons.length;
                var ts=0,ti=0;
                d.persons.forEach(function(p){ts+=p.total_standing||0;ti+=p.total_sitting||0});
                document.getElementById('ts').textContent=f(ts);
                document.getElementById('ti').textContent=f(ti);
                var t=ts+ti;
                if(t>0){document.getElementById('rg').style.width=(ts/t*100)+'%';document.getElementById('ro').style.width=(ti/t*100)+'%'}
                var al='';
                d.persons.forEach(function(p){if(p.posture==='SITTING'&&p.duration>120)al+='<div class="ab">#'+p.id+' sitting '+f(p.duration)+'</div>'});
                document.getElementById('al').innerHTML=al;
                var h='';
                if(!d.persons.length)h='<div class="nd">No persons</div>';
                d.persons.forEach(function(p){
                    var bc=p.posture==='STANDING'?'bs':p.posture==='SITTING'?'bi':'bu';
                    var sc=p.raw_score||.5;
                    var scl=sc>.55?'var(--grn)':sc<.45?'var(--org)':'var(--t2)';
                    h+='<div class="pc">';
                    h+='<div class="ph"><span class="pid">#'+p.id+'</span><span class="bd '+bc+'">'+p.posture+'</span></div>';
                    h+='<div class="pm">'+Math.round(p.confidence*100)+'% | '+f(p.duration)+' | trans:'+p.transitions+'</div>';
                    h+='<div class="scb"><div class="scf" style="width:'+(sc*100)+'%;background:'+scl+'"></div></div>';
                    h+='<div style="display:flex;justify-content:space-between;font-size:9px;color:var(--t2)"><span>SIT</span><span>'+sc.toFixed(2)+'</span><span>STAND</span></div>';
                    h+='<div style="margin-top:4px;font-size:10px;color:var(--t2)">';
                    h+='Stand: <span style="color:var(--grn)">'+f(p.total_standing)+'</span> | ';
                    h+='Sit: <span style="color:var(--org)">'+f(p.total_sitting)+'</span></div>';
                    if(p.features){
                        h+='<div style="margin-top:4px;font-size:9px;color:var(--t2);font-family:monospace">';
                        h+='AR:'+p.features.aspect_ratio+' RH:'+p.features.rel_height+'</div>';
                    }
                    h+='</div>';
                });
                document.getElementById('pp').innerHTML=h;
            }).catch(function(){});
        }
        setInterval(u,1000);
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

@app.route('/api/data')
def api_data():
    with data_lock:
        return json.dumps(frame_data)

if __name__ == '__main__':
    print("=" * 40)
    print("  POSTURE DETECTION (Ultra-Light)")
    print("  http://<jetson-ip>:5000")
    print("  Ctrl+C to stop")
    print("=" * 40)
    app.run(host='0.0.0.0', port=5000, threaded=True)
