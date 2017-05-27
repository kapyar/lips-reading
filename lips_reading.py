# -*- coding: utf-8 -*-
# USAGE
# python lips_reading.py --shape-predictor shape_predictor_68_face_landmarks.dat --video videos/*.mov --data data/aspect.dat -m false
# python lips_reading.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python lips_reading.py --video words.mov --data data/dictionary_mean.dat -m true
# python lips_reading.py --video videos/*.mov --data data/data_mean.dat --type spacial -m true
# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import feature_extractor as fe
import words_finder as wf
import math_utils as mutils
import data_loader as dl


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
    help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, required=True,
    help="path to input video file")
ap.add_argument("-d", "--data", type=str, required=True,
    help="path the vocabulary to use")
ap.add_argument("-t", "--type", type=str, required=True,
    help="<aspect | spacial>")
ap.add_argument("-m", "--multi", type=str, default="false",
    help="find start and end of the word in the stream")
args = vars(ap.parse_args())


MOUTH_AR_THRESH = 2
MOUTH_AR_CONSEC_FRAMES = 15
MOUTH_THRESH_MOVMENT = 0.3
FRAME_POOL_COUNTER = 0
MIN_FRAMES_COUNT = 6

previousAR = [0] * MOUTH_AR_CONSEC_FRAMES

# initialize the frame counters and the total number of blinks
WORD_COUNTER = 0
IS_SPEAKING = False

dl.show_dictionary(args["data"])

# print ("min len: {}".format(dl.min_word_len_in(args["data"])))

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()

if args["video"] is '':
    fileStream = False
    vs = VideoStream(src=0).start()
else:
    fileStream = True

time.sleep(1.0)

pronounced_word = list()
founded_word = "dummy"
# loop over frames from the video stream
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

        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # check if user is speaking by comparing MOUTH_AR_CONSEC_FRAMES last
        # frames from the stream if one difference is > MOUTH_THRESH_MOVMENT
        # that means user is speaking

        if args["type"] == "aspect":
            mouth_feature = fe.mouth_aspect_ratio(mouth)
        elif args["type"] == "spacial":
            mouth_feature = mouth
        else:
            print ("[ERROR] no suitable param <aspect | spacial>")

        # in case of speaking into the wild
        if args["multi"] == "true":
            if FRAME_POOL_COUNTER == MOUTH_AR_CONSEC_FRAMES:
                FRAME_POOL_COUNTER = 0

            previousAR[FRAME_POOL_COUNTER] = mouth_feature
            FRAME_POOL_COUNTER += 1

            for ar in previousAR:
                if abs(mouth_feature - ar) > MOUTH_THRESH_MOVMENT:
                    IS_SPEAKING = True
                    pronounced_word.append(mouth_feature)
                    break
                else:
                    IS_SPEAKING = False
                    if len(pronounced_word) > MIN_FRAMES_COUNT:
                        # print("word is {}".format(pronounced_word))
                        # founded_word = find_word(pronounced_word)
                        # founded_word = wf.find_by_mean(pronounced_word, args["data"])
                        if args["type"] == "aspect":
                            founded_word = wf.find_by_mean_min_max_removed(pronounced_word, args["data"])
                            print ("word: {}, mean {}".format(founded_word, mutils.mean(pronounced_word)))
                        elif args["type"] == "spacial":
                            founded_word = wf.find_special_by_frame_mean(pronounced_word, args["data"])
                            print ("word: {}, mean {}".format(founded_word, fe.mouth_mean_frame(pronounced_word)))

                    pronounced_word = []
        else:
            pronounced_word.append(mouth_feature)

        cv2.putText(frame, "Speaking: {}".format(IS_SPEAKING), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "AR: {:.2f}".format(mouth_feature), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if not IS_SPEAKING:
            cv2.putText(frame, "Word: {}".format(founded_word), (150, 250),
                cv2.FONT_ITALIC, 0.7, (0, 255, 0), 2)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

if args["multi"] == "false":
    if args["type"] == "aspect":
        founded_word = wf.find_by_mean(pronounced_word, args["data"])
        print ("[RESULT] word: {}, mean {} expected {}".format(founded_word, mutils.mean(pronounced_word), args["video"]))
    elif args["type"] == "spacial":
        founded_word = wf.find_special_by_frame_mean(pronounced_word, args["data"])
        print ("[RESULT] word spacial: {}, mean {}".format(founded_word, fe.mouth_mean_frame(pronounced_word)))

# if the `q` key was pressed, break from the loop


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()