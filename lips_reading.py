# USAGE
# python lips_reading.py --shape-predictor shape_predictor_68_face_landmarks.dat --video videos/*.mov --data data/aspect.dat -m false
# python lips_reading.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import feature_extractor as fe
import data_loader as dl
from scipy.stats import pearsonr



def find_word(feature_list):
	data = dl.load(args["data"])

	for key in data.keys():
		for word in data[key]:
			if len(word) == len(feature_list):
				if np.allclose(feature_list, word, 0.2):
					print("[INFO] founded {}".format(key))
					return key
			else:
				# subarray all available with smallest len
				if len(word) < len(feature_list):
					small = word
					big = feature_list
				else:
					small = feature_list
					big = word

				for i in range(0, (len(big) - len(small))):
					candidate = big[i : (len(small)+i)]
					if np.allclose(candidate, small, 0.2):
						print("[INFO] founded {}".format(key))
						return key


	return "Unknown"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
ap.add_argument("-d", "--data", type=str, default="",
	help="path the vocabulary to use")
ap.add_argument("-m", "--multi", type=str, default="",
	help="find start and end of the word in the stream")
args = vars(ap.parse_args())


# define two constants, one for the eye aspect ratio to indicate
# movements and then a second constant for the number of consecutive
# frames the mouth must be at the same position below the threshold

MOUTH_AR_THRESH = 2
MOUTH_AR_CONSEC_FRAMES = 15
MOUTH_THRESH_MOVMENT = 0.5
FRAME_POOL_COUNTER = 0

previousAR = [0] * MOUTH_AR_CONSEC_FRAMES

# initialize the frame counters and the total number of blinks
WORD_COUNTER = 0
IS_SPEAKING = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
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

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		mouth = shape[lStart:lEnd]

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		mouthHull = cv2.convexHull(mouth)

		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

		# check if user is speaking by comparing MOUTH_AR_CONSEC_FRAMES last
		# frames from the stream if one difference is > MOUTH_THRESH_MOVMENT
		# that means user is speaking

		mouthAR = fe.mouth_aspect_ratio(mouth)


		# in case of speaking into the wild
		if args["multi"] == "true":
			if FRAME_POOL_COUNTER == MOUTH_AR_CONSEC_FRAMES:
				FRAME_POOL_COUNTER = 0

			previousAR[FRAME_POOL_COUNTER] = mouthAR
			FRAME_POOL_COUNTER += 1

			for ar in previousAR:
				if abs(mouthAR - ar) > MOUTH_THRESH_MOVMENT:
					IS_SPEAKING = True
					pronounced_word.append(mouthAR)
					break
				else:
					IS_SPEAKING = False
					if len(pronounced_word) > 5:
						print("word is {}".format(pronounced_word))
						founded_word = find_word(pronounced_word)
					pronounced_word = []
		else:
			pronounced_word.append(mouthAR)

		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		cv2.putText(frame, "Speaking: {}".format(IS_SPEAKING), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(mouthAR), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		if not IS_SPEAKING:
			cv2.putText(frame, "Word: {}".format(founded_word), (150, 250),
				cv2.FONT_ITALIC, 0.7, (0, 255, 0), 2)

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

print("Pronounced: {}".format(pronounced_word))
print ("Finded word: {}".format(find_word(pronounced_word)))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()