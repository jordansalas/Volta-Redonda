import cv2
import numpy as np
import sys, datetime
import kairos
import base64
import threading

class FaceTracker():
    def __init__(self, frame, face):
        (x,y,w,h) = face
        self.face = (x,y,w,h)
        # Arbitrarily picked KCF tracking
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, self.face)

    def update(self, frame):
        _, self.face = self.tracker.update(frame)
        return self.face

class Controller():
    def __init__(self, event_interval=6):
        self.event_interval = event_interval
        self.last_event = datetime.datetime.now()

    def trigger(self):
        # Return True if should trigger event
        return self.get_seconds_since() > self.event_interval

    def get_seconds_since(self):
        current = datetime.datetime.now()
        seconds = (current - self.last_event).seconds
        return seconds

    def reset(self):
        self.last_event = datetime.datetime.now()

class Pipeline():
    def __init__(self, event_interval=6):
        self.controller = Controller(event_interval=event_interval)
        self.detector = FaceDetection()
        self.trackers = []

    def detect_and_track(self, frame):
        # get faces
        faces = self.detector.detect_faces_haarcascade(frame)
        #faces = self.detector.detect_faces_dnn(frame)
        # reset timer
        self.controller.reset()
        # get trackers
        self.trackers = [FaceTracker(frame, face) for face in faces]
        # return state = True for new boxes
        # if no faces detected, faces will be a tuple.
        new = type(faces) is not tuple
        return faces, new

    def track(self, frame):
        boxes = [t.update(frame) for t in self.trackers]
        # return state = False for existing boxes only
        return boxes, False

    def boxes_for_frame(self, frame):
        if self.controller.trigger():
            return self.detect_and_track(frame)
        else:
            return self.track(frame)

class FaceDetection():
    def __init__(self, file_path=None, save_path=None):
        self.file_path = file_path
        self.save_path = save_path
        self.haarcascade = '../haarcascades_cuda/haarcascade_frontalface_default.xml'
        #self.model_file = '../models/opencv_face_detector_uint8.pb'
        #self.config_file = '../models/opencv_face_detector.pbtxt'
        self.model_file = '../models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
        self.config_file = '../models/deploy.prototxt'
        #self.net = cv2.dnn.readNetFromTensorflow(self.model_file, self.config_file)
        self.net = cv2.dnn.readNetFromCaffe(self.config_file, self.model_file)
        self.red = (0, 0, 255)
        self.green = (0, 255, 0)
        self.blue = (255, 0, 0)
        self.conf_threshold = 0.7
        self.API = kairos.kairos()
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def draw_boxes(self, frame, boxes, color=(0, 255, 0)):
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
        return frame

    def detect_faces_haarcascade(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(self.haarcascade)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces

    def detect_faces_dnn(self, frame):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

        self.net.setInput(blob)
        detections = self.net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                w = abs(x2-x1)
                h = abs(y2-y1)
                bboxes.append([x1, y1, w, h])
        return bboxes

    def run_detector(self, event_interval=1):
        video_capture = cv2.VideoCapture(self.file_path)
        #video_capture = cv2.VideoCapture(0)
        _, frame = video_capture.read()
        height, width, channels = frame.shape
        out = cv2.VideoWriter(self.save_path,self.fourcc, 25.0, (width,height))

        if not video_capture.isOpened():
            print('[ERROR] Cannot open video')
            sys.exit()
        # init detection pipeline
        pipeline = Pipeline(event_interval=event_interval)
        faces = ()
        detected = False
        print("Detecting...")

        while True:
            _, frame = video_capture.read()
            boxes, detected_new = pipeline.boxes_for_frame(frame)
            if detected_new:
                print('[FACE]')
                # send fram to API recognition
                #buffer = cv2.imencode('.jpg', frame)[1].tostring()
                #jpg_as_text = base64.b64encode(buffer).decode('ascii')
                #responde = self.API.Detect(jpg_as_text)
                #t = threading.Thread(target=self.API.Detect, args=(jpg_as_text,))
                #t.start()
                color = self.green
            else:
                color = self.blue

            self.draw_boxes(frame, boxes, color)

            out.write(frame)
            cv2.imshow('Video', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

class FaceIdentification():
    def __init__(self, API):
        self.thread = threading.Thread
        if API=='kairos':
            self.API = kairos.kairos()

    def register_person(self, person_name, group_name, image_file):
        file = open(image_file, "r", encoding = "ISO-8859-1")
        for image_url in file:
            response = self.API.Enroll(image_url[:-1], group_name, person_name)
            print('Response: ', response)
            print('--------------------')

    def test(self):
        file = '00000041.jpg'
        a = True
        with open(file, 'rb') as fp:
            image = base64.b64encode(fp.read()).decode('ascii')
        print('type: ', type(image))
        print('len: ', len(image))
        print('first: ', image[:80])
        print('---')
        img = cv2.imread(file)
        buffer = cv2.imencode('.jpg', img)[1].tostring()
        jpg_as_text = base64.b64encode(buffer).decode('ascii')
        print('type: ', type(jpg_as_text))
        print('len: ', len(jpg_as_text))
        print('first: ', jpg_as_text[:80])

        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        response = self.API.Detect(jpg_as_text)
        print('Res: ', response)


def main():
    print('main')
    #FD = FaceDetection('video_1.mp4', 'detection/faces_video_1.mp4')
    FD = FaceDetection('video_2.mp4', 'detection/faces_video_2.mp4')
    FD.run_detector()

    #FI = FaceIdentification('kairos')
    #FI.register_person("alan_grant", "Dj", "data_set/alan_grant.txt")
    #FI.test()


if __name__ == "__main__":
    main()
