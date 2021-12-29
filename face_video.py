import face_recognition
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--videos", type=str, required=True, help="path to input video")
ap.add_argument("-i", "--images", type=str, required=True, help="path to images directory of cropped faces")
ap.add_argument("-d", "--detection-method", type=str, default="hog", help="face detection model to use: either 'hog' or 'cnn'")
ap.add_argument("-s", "--skip", type=int, default=16, help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

vs = cv2.VideoCapture(args["videos"])
read = 0
saved = 0

while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break

    read += 1
    if read % args["skip"] != 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    
    for (top, right, bottom, left) in boxes:
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        p = os.path.sep.join([args["images"], "{}.png".format(saved+1)])
        saved += 1

        face = frame[top:bottom, left:right]
        face = imutils.resize(face, width=300)
        cv2.imwrite(p, face)

        print("Saved {} images to disk".format(p))

vs.release()
cv2.destroyAllWindows()