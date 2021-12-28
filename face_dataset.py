from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import shutil
import time
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detection-method", type=str, default="hog", help="face detection model to use: either 'hog' or 'cnn'")
ap.add_argument("-i", "--dataset", required=True, help="path to input directory of faces + images")
args = vars(ap.parse_args())

inp_name = input("Please insert your name: ")
path_name = os.path.join(args["dataset"], inp_name.replace(" ", "-").lower())

print("Creating path...")
for dir in next(os.walk(args["dataset"]))[1]:
    name = os.path.basename(path_name)

    if dir == name:
        for file in next(os.walk(path_name))[2]:
            file_path = os.path.join(path_name, file)

            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)

        os.rmdir(path_name)

os.mkdir(path_name)

print("Starting video stream...")
vs = VideoStream(src=4).start() # integrated = 0, external = 4
time.sleep(2.0) # warm-up camera

total = 0

while True:
    frame = vs.read()
    frame_copy = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])

    for (top, right, bottom, left) in boxes:
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        text = "Captured photos: {} of 30".format(total)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("k"):
        total += 1
        p = os.path.sep.join([path_name, "{}.png".format(str(total).zfill(5))])
        
        frame_copy = imutils.resize(frame_copy, height=720)
        cv2.imwrite(p, frame_copy)
        
    if key == ord("q"):
        break

    if total == 30:
        break

print("{} face images stored".format(total))
print("Cleaning up...")

cv2.destroyAllWindows()
vs.stop()