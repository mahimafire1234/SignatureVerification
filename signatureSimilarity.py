import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from cv2 import  *
#displaying the signature
image_on_cheque = cv2.imread("images/harrysig.png")
image_on_software= cv2.imread("images/harrysign1.png")

#resizing image
image_on_cheque_resize = imutils.resize(image_on_cheque,height=300)
image_on_software_resize = imutils.resize(image_on_software,height=300)

#grayscaling the image
gray_scaled_image = cv2.cvtColor(image_on_cheque_resize,cv2.COLOR_BGRA2GRAY)
gray_Scaled_cheque = cv2.cvtColor(image_on_software_resize,cv2.COLOR_BGRA2GRAY)
#removing noise by using Gaussian blur
gray_scaled_image = cv2.GaussianBlur(gray_scaled_image,(5,5),0)
gray_Scaled_cheque = cv2.GaussianBlur(gray_Scaled_cheque,(5,5),0)

edged_image_cheque = cv2.Canny(gray_scaled_image,100,200)
edged_image_software = cv2.Canny(gray_Scaled_cheque,100,200)

#ORB Detector
orb = cv2.ORB_create()
# key points and descriptor calculation
kp1, cheque_image = orb.detectAndCompute(edged_image_cheque, None)
kp2, software_image = orb.detectAndCompute(edged_image_software, None)
# creating matches
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
matches_1 = matcher.knnMatch(cheque_image, software_image, 2)
len(matches_1)

#distance similarity
good_points = []
for m,n in matches_1:
    if m.distance < 0.89* n.distance:
        good_points.append(m)
len(good_points)

result = cv2.drawMatches(cheque_image, kp1 , software_image, kp2, good_points, None)
#calculating ratio
print("The match points are : ",len(good_points))
if(len(good_points)>120):
    print("The transaction can be proceeded.")
else:
    print("The transaction cannot be proceeded.")


#wait key
# cv2.waitKey(0)
# cv2.destroyAllWindows()