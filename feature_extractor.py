from scipy.spatial import distance as dist


def mouth_aspect_ratio(mouth_frame):
    # distance between outer contoure of lisps
    width  = dist.euclidean(mouth_frame[0], mouth_frame[6])
    height = dist.euclidean(mouth_frame[3], mouth_frame[9])

    # compute the mouth aspect ratio
    ration = width/height

    return ration


# result list of float
def spacial_centroid(word):
    mean_frames = list()
    for frame in word:
        mean_frames.append(sum(frame) / float(len(frame)))
    mean_word = sum(mean_frames) / float(len(mean_frames))

    return mean_word


# as result 20 points
def spacial_mean_points(word):
    sumator = word[0]
    for frame in word[1:]:
        sumator += frame

    return sumator/float(20)
