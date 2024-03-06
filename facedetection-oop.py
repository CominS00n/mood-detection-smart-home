import argparse
import os
import platform
import sys
from pathlib import Path

import torch

import cv2
import numpy as np
from keras.models import model_from_json
from yeelight import Bulb, discover_bulbs
import time

import threading

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2, increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh
from utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
class FaceDetector:
    def __init__(self, weights, source, data, imgsz, conf_thres, iou_thres, max_det, device, classes, agnostic_nms, augment, visualize, update, line_thickness, hide_labels, hide_conf, half, dnn, vid_stride):
        self.weights = weights
        self.source = source
        self.data = data
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = False
        self.save_crop = False
        self.nosave = False
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.update = update
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half
        self.dnn = dnn
        self.vid_stride = vid_stride
        # self.stop_webcam = False

    @smart_inference_mode()
    def run(self):
        source = str(self.source)
        save_img = not self.nosave and not source.endswith('.txt')
        is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = str(self.source).isnumeric() or str(self.source).endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')

        if is_url and is_file:
            source = check_file(source)

        device = select_device(self.device)
        model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(self.imgsz, s=stride)

        bs = 1
        if webcam:
            self.view_img = check_imshow(warn=True)
            dataset = LoadStreams(self.source, img_size=imgsz, stride=stride, auto=pt, vid_stride=self.vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(self.source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride, auto=pt, vid_stride=self.vid_stride)

        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        # start_time = time.time()
        # duration = 3
        for path, im, im0s, vid_cap, s in dataset:
            # if self.stop_webcam or time.time() - start_time >= duration:
            #     self.cap.release()
            #     break
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()
                im /= 255
                if len(im.shape) == 3:
                    im = im[None]

            with dt[1]:
                pred = model(im, augment=self.augment, visualize=self.visualize)

            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            for i, det in enumerate(pred):
                seen += 1
                if webcam:
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)
                s += '%gx%g ' % im.shape[2:]
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        label = names[c] if self.hide_conf else f'{names[c]}'

                        if names[int(c)] == 'visitor':
                            print("Visitor detected! Take appropriate action.")
                        else:
                            # self.stop_webcam = True
                            emotion_detector = EmotionDetector()
                            emotion_detector.run(duration=3)
                            emotion_detector.process_emotion_array()
                            # break

                        if save_img or self.save_crop or self.view_img:
                            c = int(cls)
                            label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))

                im0 = annotator.result()
                if self.view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                        cv2.resizeWindow(str(p), im0.shape[0], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)

            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        cv2.destroyAllWindows()

        t = tuple(x.t / seen * 1E3 for x in dt)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if self.update:
            strip_optimizer(self.weights[0])

class EmotionDetector:
    max_emotion_array = []

    def __init__(self):
        self.emotion_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}
        self.load_emotion_model()
        self.cap = cv2.VideoCapture(0)
        self.bulb = Bulb("192.168.1.119")  # Replace with your bulb's IP address
        self.emotion_array_list = []
        self.start_time = time.time()
    
    def load_emotion_model(self):
        json_file = open('models/emotion/emotion_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.emotion_model = model_from_json(loaded_model_json)
        self.emotion_model.load_weights("models/emotion/emotion_model.h5")
        print("Loaded model from disk")

    def detect_emotion(self, frame, duration):
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = self.emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            self.emotion_array_list.append(emotion_prediction[0].tolist())

            elapsed_time = time.time() - self.start_time

            if elapsed_time >= duration:
                max_emotion_array = max(self.emotion_array_list, key=lambda x: x[0])
                maxindex = max_emotion_array.index(max(max_emotion_array))

                # ทำงานกับ max_emotion_array ตามที่คุณต้องการ
                self.process_emotion_array(max_emotion_array)

                # รีเซ็ตค่า emotion_array_list และ start_time เพื่อให้เริ่มนับใหม่ในรอบถัดไป
                self.emotion_array_list = []
                self.start_time = time.time()
            cv2.putText(frame, self.emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        return frame
    
    def process_emotion_array(self, max_emotion_array):
        maxindex = max_emotion_array.index(max(max_emotion_array))
        emotion_label = self.emotion_dict[maxindex]
        if emotion_label == "Angry":
            self.set_bulb_color(41, 197, 246)  # Blue
            print("open blue light #29C5F6")
        elif emotion_label == "Happy":
            self.set_bulb_color(230, 126, 34)  # Orange
            print("open orange light #E67E22")
        elif emotion_label == "Neutral":
            self.bulb.turn_off()
            print("off light")
        elif emotion_label == "Sad":
            self.set_bulb_color(39, 174, 96)  # Green
            print("open green light #27AE60")
        elif emotion_label == "Surprised":
            self.set_bulb_color(244, 208, 63)  # Yellow
            print("open yellow light #F4D03F")
        else:
            print(f"Unknown emotion label: {emotion_label}")

    def set_bulb_color(self, r, g, b):
        print(f"Setting bulb color: R={r}, G={g}, B={b}")
        self.bulb.turn_on()
        self.bulb.set_rgb(r, g, b)
        self.bulb.set_brightness(50)

    def run(self, duration=3):
        start_time = time.time()

        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, (720, 480))
            if not ret:
                break

            frame = self.detect_emotion(frame, duration)
            cv2.imshow('Emotion Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        max_emotion_array = max(self.emotion_array_list, key=lambda x: x[0])
        self.process_emotion_array(max_emotion_array)

    


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'vMembers_v2.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    faceDetector = FaceDetector(opt.weights, opt.source, opt.data, opt.imgsz, opt.conf_thres, opt.iou_thres, opt.max_det, opt.device, opt.classes, opt.agnostic_nms, opt.augment, opt.visualize, opt.update, opt.line_thickness, opt.hide_labels, opt.hide_conf, opt.half, opt.dnn, opt.vid_stride)
    faceDetector.run()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
