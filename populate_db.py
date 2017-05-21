# -*- coding: utf-8 -*-
# USAGE
# python populate_db.py --origin videos --destination data/dictionary --shape-predictor shape_predictor_68_face_landmarks.dat --type aspect --video false

#

from imutils.video import FileVideoStream
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
from os import listdir
from os.path import isfile, join
import feature_extractor as fe
import pickle
import time
from collections import defaultdict


def populate_aspect_ratio():
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    allfiles = [f for f in listdir(args["origin"]) if isfile(join(args["origin"], f))]

    words = defaultdict(list)

    for file in allfiles:
        if file.endswith(".mov"):
            # start the video stream thread
            file_name = file[:-4]
            print ("Filename: {}".format(file_name))

            print("[INFO] starting video stream thread... file: {}".format(join(args["origin"], file)))
            vs = FileVideoStream(join(args["origin"], file)).start()

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
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


                # detect faces in the grayscale frame
                rects = detector(gray, 0)
                # loop over the face detections
                for rect in rects:
                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    # extract the left and right eye coordinates, then use the
                    # coordinates to compute the eye aspect ratio for both eyes
                    mouth = shape[lStart:lEnd]

                    # compute the convex hull for the mouth
                    if args["video"] is "true":
                        mouthHull = cv2.convexHull(mouth)
                        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

                    # check if user is speaking by comparing MOUTH_AR_CONSEC_FRAMES last
                    # frames from the stream if one difference is > MOUTH_THRESH_MOVMENT
                    # that means user is speaking

                    mouthAR = fe.mouth_aspect_ratio(mouth)

                    pronounced_word.append(mouthAR)

                    # draw the log
                    # show the frame
                    if args["video"] is "true":
                        cv2.putText(frame, "EAR: {:.2f}".format(mouthAR), (300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if args["video"] is "true":
                    cv2.imshow("Frame", frame)

            words[file_name].append(pronounced_word)

            # do a bit of cleanup
            cv2.destroyAllWindows()
            vs.stop()

    # save into destination


    with open("{}{}".format(args["destination"], "_aspect.dat"), 'wb') as handle:
        pickle.dump(words, handle, protocol=pickle.HIGHEST_PROTOCOL)

    mean = defaultdict(list)
    for key in words.keys():
        word_features = words[key]
        for word in word_features:
            print ("word: {}".format(word))
            mean[key].append(sum(word)/float(len(word)))

    print("Mean values: {}".format(mean))

    with open("{}{}".format(args["destination"], "_mean.dat"), 'wb') as handle:
            pickle.dump(mean, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open(args["destination"], 'rb') as handle:
    #     b = pickle.load(handle)
    #
    # print("Check serialization {}".format(words == b))





# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--origin", required=True,
	help="path to the raw video")

ap.add_argument("-d", "--destination", required=True, type=str, default="",
	help="path to store extracted features")

ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")

ap.add_argument("-t", "--type", type=str, default="aspect",
	help="aspect based or full points approach params: aspect, dots")

ap.add_argument("-v", "--video", type=str, default="false",
	help="is need to show video when extracting features")

args = vars(ap.parse_args())
print("[INFO] args: {}".format(args))

if args["type"] == "aspect":
    populate_aspect_ratio()
else:
    print ("[INFO] not aspect")

