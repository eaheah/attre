# USAGE
# python align_faces.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from face_aligner import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
from matplotlib import pyplot as plt


# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread('images/example_01.jpg')
image = imutils.resize(image, width=800)
RGB_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# show the original input image and detect faces in the grayscale
# image
# cv2.imshow("Input", image)
rects = detector(gray, 2)

# loop over the face detections
for rect in rects:
    # extract the ROI of the *original* face, then align the face
    # using facial landmarks
    (x, y, w, h) = rect_to_bb(rect)
    faceOrig = imutils.resize(RGB_im[y:y + h, x:x + w], width=256)
    faceAligned = fa.align(image, gray, rect)
    faceAligned_RGB =  cv2.cvtColor(faceAligned, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15,15))

    plt.subplot(121)
    plt.imshow(faceOrig)

    plt.subplot(122)

    plt.imshow(faceAligned_RGB)
    plt.savefig('img1.png')