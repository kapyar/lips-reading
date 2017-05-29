# -*- coding: utf-8 -*-
# USAGE
# python populate_db.py --origin videos --destination data/dictionary --type aspect --size 5
# python populate_db.py --origin videos --destination data/dictionary --type special --size 5

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
import consts


def populate_aspect_ratio():
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    allfiles = [f for f in listdir(args["origin"]) if isfile(join(args["origin"], f))]

    print (listdir(args["origin"]))

    words = defaultdict(list)

    for dir in listdir(args["origin"]):

        if dir == ".DS_Store":
            continue

        number_of_added_words = 0

        for file in listdir(join(args["origin"], dir)):
            if isfile(join(args["origin"], join(dir, file))):
                if file.endswith(".mov"):
                    # start the video stream thread
                    file_name = file[:-4]

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
                        for rect in rects:
                            # determine the facial landmarks for the face region, then
                            # convert the facial landmark (x, y)-coordinates to a NumPy
                            # array
                            shape = predictor(gray, rect)
                            shape = face_utils.shape_to_np(shape)

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
                                cv2.putText(frame, "AR: {:.2f}".format(mouthAR), (300, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        if args["video"] is "true":
                            cv2.imshow("Frame", frame)

                    words[dir].append(pronounced_word)
                    number_of_added_words += 1

                    if number_of_added_words >= args["size"]:
                        print ("go to new word")
                        break

                    # do a bit of cleanup
                    cv2.destroyAllWindows()
                    vs.stop()

    # save into destination


    with open("{}{}".format(args["destination"], consts.ASPECT), 'wb') as handle:
        pickle.dump(words, handle, protocol=pickle.HIGHEST_PROTOCOL)

    mean = defaultdict(list)
    for key in words.keys():
        word_features = words[key]
        for word in word_features:
            mean[key].append(sum(word) / float(len(word)))

    with open("{}{}".format(args["destination"], consts.ASPECT_MEAN), 'wb') as handle:
        pickle.dump(mean, handle, protocol=pickle.HIGHEST_PROTOCOL)


def populate_spacial():
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # grab the indexes of the facial landmarks for the mouth
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    allfiles = [f for f in listdir(args["origin"]) if isfile(join(args["origin"], f))]

    print (listdir(args["origin"]))

    words = defaultdict(list)

    for dir in listdir(args["origin"]):

        if dir == ".DS_Store":
            continue

        number_of_added_words = 0

        for file in listdir(join(args["origin"], dir)):
            if isfile(join(args["origin"], join(dir, file))):
                if file.endswith(".mov"):
                    # start the video stream thread
                    file_name = file[:-4]

                    print("[INFO] starting video stream thread... file: {}".format(join(args["origin"], join(dir, file))))
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
                        for rect in rects:
                            # determine the facial landmarks for the face region, then
                            # convert the facial landmark (x, y)-coordinates to a NumPy
                            # array
                            shape = predictor(gray, rect)
                            shape = face_utils.shape_to_np(shape)

                            mouth = shape[lStart:lEnd]

                            pronounced_word.append(mouth)

                        if args["video"] is "true":
                            cv2.imshow("Frame", frame)

                    words[dir].append(pronounced_word)

                    number_of_added_words += 1

                    if number_of_added_words >= args["size"]:
                        print ("go to new word")
                        break

                    # do a bit of cleanup
                    cv2.destroyAllWindows()
                    vs.stop()

    # save into destination

    with open("{}{}".format(args["destination"], consts.SPACIAL), 'wb') as handle:
        pickle.dump(words, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # calculate a little bit different
    # mean of mouth position on each frame
    mean = defaultdict(list)
    for key in words.keys():
        for word in words[key]:  # [[x, y],[x, y]..[]] == length in frames of word each frame consists of 20 params [x, y]
            mean_word = fe.spacial_centroid(word)
            mean[key].append(mean_word)

    with open("{}{}".format(args["destination"], consts.SPACIAL_CENTROID), 'wb') as handle:
        pickle.dump(mean, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # calculate a little bit different
        # mean of mouth position on each frame
        points_mean_shift = defaultdict(list)
        for key in words.keys():
            for word in words[key]:
                mean_word = fe.spacial_mean_points(word)
                points_mean_shift[key].append(mean_word)

    with open("{}{}".format(args["destination"], consts.SPACIAL_MEAN_SHIFT), 'wb') as handle:
        pickle.dump(points_mean_shift, handle, protocol=pickle.HIGHEST_PROTOCOL)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--origin", required=True,
	help="path to the raw video")

ap.add_argument("-d", "--destination", required=True, type=str, default="",
	help="path to store extracted features")

ap.add_argument("-s", "--size", required=True, type=int, default="",
	help="size of trained data")

ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
	help="path to facial landmark predictor")

ap.add_argument("-t", "--type", type=str, default="aspect",
	help="aspect based or full points approach params: <aspect | special>")

ap.add_argument("-v", "--video", type=str, default="false",
	help="is need to show video when extracting features")

args = vars(ap.parse_args())
print("[INFO] args: {}".format(args))

if args["type"] == "aspect":
    populate_aspect_ratio()
else:
    populate_spacial()

