import cv2
from skimage.metrics import structural_similarity
# for same sized images
def check_similarity(image1,image2):
    similarity,difference = structural_similarity(image1,image2,full=True)
    return similarity

#for different sized images
def check_similarity_different_sizes(image1,image2):
    orb = cv2.ORB_create()

    # detect keypoints and features
    keypoint_a, features_a = orb.detectAndCompute(image1,None)
    keypoint_b, features_b = orb.detectAndCompute(image2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(features_a, features_b)
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)

img00 = cv2.imread('images/signature1.png', 0)
img01 = cv2.imread('images/signature2.png', 0)

img1 = cv2.imread('images/BSE.jpg', 0)  # 714 x 901 pixels
img2 = cv2.imread('images/BSE_noisy.jpg', 0)  # 714 x 901 pixels
img3 = cv2.imread('images/BSE_smoothed.jpg', 0)  # 203 x 256 pixels
img4 = cv2.imread('images/different_img.jpg', 0)  # 203 x 256 pixels

# checking similairty
orb_similarity = check_similarity_different_sizes(img00, img01)  #1.0 means identical. Lower = not similar
print("Similarity amongst the signatures are : ", orb_similarity*100)

