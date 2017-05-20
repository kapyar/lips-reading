from scipy.spatial import distance as dist



def mouth_aspect_ratio(mouth):
    # distance between outer contoure of lisps
    width  = dist.euclidean(mouth[0], mouth[6])
    height = dist.euclidean(mouth[3], mouth[9])

    # compute the eye aspect ratio
    ration = width/height

    return ration