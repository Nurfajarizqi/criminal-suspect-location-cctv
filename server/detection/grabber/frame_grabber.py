import cv2, time, threading, queue
from config import GREEN, RED, RESET

class FrameGrabber(threading.Thread):
    def __init__(self, url, q, stop_evt):
        super().__init__(daemon=True)
        self.url, self.q, self.stop_evt = url, q, stop_evt
        self.cap, self.prev_time, self.fps = None, time.time(), 0.0

    def _open(self):
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(self.url)
        print(f"{GREEN}[STATUS]{RESET} Stream opened" if self.cap.isOpened() else f"{RED}[ERROR]{RESET} Failed to open stream")

    def run(self):
        self._open()
        while not self.stop_evt.is_set():
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.5)
                self._open()
                continue
            ret, frame = self.cap.read()
            if not ret: time.sleep(0.1); continue
            ts = time.time()
            self.fps = 1.0 / (ts - self.prev_time + 1e-10)
            self.prev_time = ts
            try:
                self.q.put_nowait((frame, ts))
            except queue.Full:
                try: self.q.get_nowait(); self.q.put_nowait((frame, ts))
                except queue.Empty: pass
        if self.cap: self.cap.release()
        print(f"{RED}[INFO]{RESET} FrameGrabber stopped.")