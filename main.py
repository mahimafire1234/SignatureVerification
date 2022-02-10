from cv2 import *

#uploading images
template = cv2.imread("images/signature1.png")
original = cv2.imread("images/signature3.png")

#resizing images
template = cv2.resize(template,(528,152))
cv2.imshow("template image", template)
cv2.waitKey(0)
cv2.destroyAllWindows()
template.shape #row.columns
original = cv2.resize(original,(528,152))
cv2.imshow("original image", original)
cv2.waitKey(0)
cv2.destroyAllWindows()

#ORB Detector
orb = cv2.ORB_create()

original = cv2.Canny(original, 50, 200)
template = cv2.Canny(template, 50, 200)

# key points and descriptor calculation
kp1, desc_1 = orb.detectAndCompute(template, None)
kp2, desc_2 = orb.detectAndCompute(original, None)

#creating matches
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
matches_1 = matcher.knnMatch(desc_1, desc_2, 2)
len(matches_1)

result = cv2.drawMatchesKnn(original, kp1 , template, kp2, matches_1, None)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#distance similarity
good_points = []
for m,n in matches_1:
    if m.distance < 0.8* n.distance:
        good_points.append(m)
len(good_points)

result = cv2.drawMatches(original, kp1 , template, kp2, good_points, None)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(len(kp1))
print(len(kp2))

#calculating ratio
print("How good is the match : ",len(kp1)/len(good_points))