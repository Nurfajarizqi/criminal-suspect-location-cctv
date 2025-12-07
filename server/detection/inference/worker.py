import torch, time, json, gc, sys, threading, numpy as np
from config import GREEN, RED, YELLOW, RESET, YOLO_IMG_SIZ, YOLO_CONF_THRESHOLD, RECOG_CONF_THRESHOLD, MIN_PROC_INTERVAL
from inference.models import device
import cv2

def preprocess_face(img_rgb, box, size=160, stream=None):
    x1, y1, x2, y2 = map(int, box)
    h, w = img_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    face = cv2.resize(img_rgb[y1:y2, x1:x2], (size, size))
    t = torch.from_numpy(np.transpose(face, (2,0,1)).astype(np.float32)/255.0).unsqueeze(0)
    if device.type=="cuda":
        with torch.cuda.stream(stream or torch.cuda.current_stream()):
            t = t.to(device, non_blocking=True).half()
    else:
        t = t.to(device)
    return t

class InferenceWorker(threading.Thread):
    def __init__(self, q, stop_evt, grabber, detector, facenet, svm, encoder, pca=None, normalizer=None):
        super().__init__(daemon=True)
        self.q, self.stop_evt, self.grabber = q, stop_evt, grabber
        self.detector, self.facenet, self.svm = detector, facenet, svm
        self.encoder, self.pca, self.normalizer = encoder, pca, normalizer
        self.frame_count, self.rekap_data = 0, []
        self.cuda_stream = torch.cuda.Stream() if device.type=="cuda" else None
        self.prev_time, self.fps = time.time(), 0.0
        self.last_print_time, self.last_names = 0, []

    def run(self):
        print(f"{GREEN}[INFO]{RESET} InferenceWorker started")
        try:
            while not self.stop_evt.is_set():
                try:
                    frame, _ = self.q.get(timeout=0.5)
                except:
                    continue

                loop_start = time.time()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = []

                try:
                    results = self.detector.predict(source=rgb, imgsz=YOLO_IMG_SIZ, conf=YOLO_CONF_THRESHOLD, verbose=False)
                    if results and len(results) > 0:
                        r = results[0]
                        if hasattr(r, "boxes") and r.boxes:
                            boxes = r.boxes.xyxy.cpu().numpy()
                            confs = r.boxes.conf.cpu().numpy()
                            tensors, face_boxes = [], []
                            for i, box in enumerate(boxes):
                                if confs[i] < YOLO_CONF_THRESHOLD:
                                    continue
                                t = preprocess_face(rgb, box, stream=self.cuda_stream)
                                if t is not None:
                                    tensors.append(t)
                                    face_boxes.append(box)

                            if tensors:
                                batch = torch.cat(tensors, dim=0)
                                with torch.no_grad():
                                    if device.type=="cuda":
                                        batch = batch.half()
                                        with torch.cuda.stream(self.cuda_stream):
                                            emb = self.facenet(batch)
                                        torch.cuda.synchronize()
                                    else:
                                        emb = self.facenet(batch)

                                emb_np = emb.detach().cpu().numpy()
                                if self.normalizer:
                                    emb_np /= (np.linalg.norm(emb_np, axis=1, keepdims=True)+1e-10)
                                if self.pca:
                                    emb_np = self.pca.transform(emb_np)

                                try:
                                    preds = self.svm.predict(emb_np)
                                except:
                                    preds = [int(self.svm.predict(e.reshape(1,-1))[0]) for e in emb_np]

                                for i, pred in enumerate(preds):
                                    pid = int(pred)
                                    try:
                                        name = self.encoder.inverse_transform([pid])[0]
                                    except:
                                        name = str(pid)
                                    if 1.0 < RECOG_CONF_THRESHOLD:
                                        name = "Unknown"
                                    detections.append({"id": pid, "name": name, "box": [int(x) for x in face_boxes[i]]})
                except Exception as e:
                    print(f"{RED}[ERROR]{RESET} Inference: {e}")

                self.frame_count += 1
                current_time = time.time()
                self.fps = 1.0/(current_time-self.prev_time+1e-10)
                self.prev_time = current_time
                names = [d["name"] for d in detections] if detections else ["-"]

                if current_time - self.last_print_time > 0.2 or names != self.last_names:
                    sys.stdout.write('\r'+' '*150)
                    timestamp = time.strftime("%H:%M:%S")
                    sys.stdout.write(f"\r[{timestamp}] {YELLOW}[STATUS]{RESET} Camera FPS:{self.grabber.fps:5.1f} | Infer FPS:{self.fps:5.1f} | Faces:{len(detections):2d} | Names:{', '.join(names)[:40]}")
                    sys.stdout.flush()
                    self.last_print_time, self.last_names = current_time, names

                self.rekap_data.append({"frame":self.frame_count, "timestamp":time.time(), "faces":detections})
                if len(self.rekap_data) >= 10:
                    with open("rekap_wajah.json","w") as f:
                        json.dump(self.rekap_data, f, indent=2)

                elapsed = time.time()-loop_start
                if elapsed < MIN_PROC_INTERVAL:
                    time.sleep(MIN_PROC_INTERVAL - elapsed)
        finally:
            self.cleanup()
            print(f"\n{RED}[INFO]{RESET} InferenceWorker stopped.")

    def cleanup(self):
        try:
            # Simpan rekap wajah terakhir
            with open("rekap_wajah.json","w") as f:
                json.dump(self.rekap_data,f,indent=2)
            # Jangan hapus SVM secara eksplisit, biarkan Python mengurus
            gc.collect()
            if device.type=="cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"{RED}[ERROR]{RESET} Cleanup failed: {e}")