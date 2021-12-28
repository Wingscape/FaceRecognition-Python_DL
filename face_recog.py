from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detection-method", type=str, default="hog", help="face detection model to use: either 'hog' or 'cnn'")
ap.add_argument("-r", "--recognizer", required=True, help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True, help="path to label encoder")
args = vars(ap.parse_args())

print("Loading face recognizer...")
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

print("Starting video stream...")
vs = VideoStream(src=4).start() # integrated = 0, external = 4
time.sleep(2.0) # warm-up camera

fps = FPS().start()

while True:
    frame = vs.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    probas = []

    for encoding in encodings:
        preds = recognizer.predict_proba([encoding])[0]

        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]
        
        probas.append(proba)
        names.append(name)
    
    for ((top, right, bottom, left), name, proba) in zip(boxes, names, probas):
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        text = "{}: {:.2f}%".format(name, proba * 100)
        y = top - 15 if top -15 > 15 else top + 15
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, text, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    fps.update()

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

fps.stop()
print("Elasped time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

print("Cleaning up...")
cv2.destroyAllWindows()
vs.stop()