from imutils import paths
import face_recognition
import numpy as np
import argparse
import imutils
import shutil
import cv2
import os

def align(image_path, width=128, height=128):
    # load image and find face locations.
    image = face_recognition.load_image_file(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb, model="hog")

    # detect 68-landmarks from image. This includes left eye, right eye, lips, eye brows, nose and chins
    face_landmarks = face_recognition.face_landmarks(rgb)

    '''
    Let's find and angle of the face. First calculate 
    the center of left and right eye by using eye landmarks.
    '''
    leftEyePts = face_landmarks[0]['left_eye']
    rightEyePts = face_landmarks[0]['right_eye']

    leftEyeCenter = np.array(leftEyePts).mean(axis=0).astype("int")
    rightEyeCenter = np.array(rightEyePts).mean(axis=0).astype("int")

    leftEyeCenter = (leftEyeCenter[0],leftEyeCenter[1])
    rightEyeCenter = (rightEyeCenter[0],rightEyeCenter[1])

    # draw the circle at centers and line connecting to them
    # cv2.circle(image, leftEyeCenter, 2, (255, 0, 0), 10)
    # cv2.circle(image, rightEyeCenter, 2, (255, 0, 0), 10)
    # cv2.line(image, leftEyeCenter, rightEyeCenter, (255,0,0), 10)

    # find and angle of line by using slop of the line.
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # to get the face at the center of the image,
    # set desired left eye location. Right eye location 
    # will be found out by using left eye location.
    # this location is in percentage.
    desiredLeftEye=(0.35, 0.35)
    #Set the croped image(face) size after rotaion.
    desiredFaceWidth = width
    desiredFaceHeight = height

    desiredRightEyeX = 1.0 - desiredLeftEye[0]
    
    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = (int((leftEyeCenter[0] + rightEyeCenter[0]) // 2), int((leftEyeCenter[1] + rightEyeCenter[1]) // 2))

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    # (y2,x2,y1,x1) = face_locations[0]
            
    output = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_CUBIC)

    return output

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ds_dir = os.path.join(BASE_DIR, "dataset_aligned")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to input directory of faces + images")
args = vars(ap.parse_args())

print("Quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

amount_pic = 0
name_temp = ""

for (i, imagePath) in enumerate(imagePaths):
    print("Processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    path_name = os.path.join(ds_dir, name)

    if name != name_temp:
        name_temp = name

        for dir in next(os.walk(ds_dir))[1]:
            if dir == name:
                for file in next(os.walk(path_name))[2]:
                    file_path = os.path.join(path_name, file)

                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)

                os.rmdir(path_name)

        os.mkdir(path_name)
    
    amount_pic += 1
    img_item = "{}.png".format(str(amount_pic).zfill(5))

    face_aligned = align(imagePath, 256, 256)
    cv2.imwrite(os.path.join(path_name, img_item), face_aligned)

print("{} face images stored".format(amount_pic))
print("Cleaning up...")