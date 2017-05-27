# -*- coding: utf-8 -*-
# USAGE
# python populate_db.py --origin videos --destination data/dictionary --shape-predictor shape_predictor_68_face_landmarks.dat --type aspect --video false
# python populate_db.py --origin videos --destination data/dictionary --shape-predictor shape_predictor_68_face_landmarks.dat --type special
# python populate_db.py --origin videos --destination data/dictionary --type special

from imutils.video import FileVideoStream
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
from os import listdir
from os.path import isfile, join
import feature_extractor as fe
import words_finder as wf
import time
from collections import defaultdict


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True,
	help="path to the data that stored in db")

ap.add_argument("-o", "--origin", required=True,
	help="video to validate")

ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
	help="path to facial landmark predictor")

ap.add_argument("-t", "--type", type=str, default="aspect",
	help="aspect based or full points approach params: <aspect | special>")

ap.add_argument("-m", "--method", type=str, default="aspect",
	help="aspect based or full points approach params: <full | mean | mean_min_max | mean_frame>")


args = vars(ap.parse_args())


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

allfiles = [f for f in listdir(args["origin"]) if isfile(join(args["origin"], f))]

print (listdir(args["origin"]))

words = defaultdict(list)
founded = 0
miss = 0
total = 0
founded_word = "Unknown"

for dir in listdir(args["origin"]):

    if dir == ".DS_Store":
        continue

    for file in listdir(join(args["origin"], dir)):
        if isfile(join(args["origin"], join(dir, file))):
            if file.endswith(".mov"):
                # start the video stream thread
                expected_word = dir
                total+=1

                print(
                "[INFO] starting video stream thread... file: {}".format(join(args["origin"], join(dir, file))))
                vs = FileVideoStream(join(args["origin"], join(dir, file))).start()

                fileStream = True

                time.sleep(2.0)
                pronounced_word = list()

                while True:
                    # if this is a file video stream, then we need to check if
                    # there any more frames left in the buffer to process

                    if fileStream and not vs.more():
                        break

                    # grab the frame from the threaded video file stream, resize
                    # it, and convert it to grayscale
                    # channels)
                    frame = vs.read()
                    frame = imutils.resize(frame, width=450)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # detect faces in the grayscale frame
                    rects = detector(gray, 0)
                    # loop over the face detections
                    print ("rects.length {}".format(len(rects)))
                    for rect in rects:
                        # determine the facial landmarks for the face region, then
                        # convert the facial landmark (x, y)-coordinates to a NumPy
                        # array
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        # extract the left and right eye coordinates, then use the
                        # coordinates to compute the eye aspect ratio for both eyes
                        mouth = shape[lStart:lEnd]

                        # todo add method choose
                        # aspect : mean, min_max_aspect
                        # help="aspect based or full points approach params: <full | mean | mean_min_max | mean_frame>")
                        if args["type"] == "aspect":
                            mouthAR = fe.mouth_aspect_ratio(mouth)
                            pronounced_word.append(mouthAR)
                        if args["type"] == "spacial":
                            pronounced_word.append(shape)

                        # add break in the end ???

                # find word from db
                if args["type"] == "aspect":
                    if args["method"] == "mean":
                        founded_word = wf.find_by_mean(pronounced_word, args["data"])
                    elif args["method"] == "mean_min_max":
                        founded_word = wf.find_by_mean_min_max_removed(pronounced_word, args["data"])
                    elif args["method"] == "full":
                        founded_word = wf.find_word_exactly(pronounced_word, args["data"])
                    else:
                        print ("[INFO] no parameter {} ".format(args["method"]))
                elif args["type"] == "spacial":
                    print ("[INFO] spacial part")


                print ("Expected: {}, actual: {}".format(expected_word, founded_word))

                # do a bit of cleanup
                cv2.destroyAllWindows()
                vs.stop()
            print ("[RESULT] founded: {}, miss: {}   {}%".format(founded, miss, founded/float(total)))