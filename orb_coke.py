import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

path1 = "EDSR/EDSR_x4.pb"
path2 = "ESPCN/ESPCN_x4.pb"
path3 = "FSRCNN/FSRCNN_x4.pb"
path4 = "LapSRN/LapSRN_x8.pb"

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(path2)
sr.setModel("espcn",4)
#sr.setModel("fsrcnn",4)
#sr.setModel("lapsrn",8)

orb = cv2.ORB_create(nfeatures=50000)
# deal with id_001 image
img1 = cv2.imread("previous_img/id_001.png")
#plt.imshow(img1[:,:,::-1])
#plt.show()
img2 = cv2.imread("current_img/id_001.png",0)

img1 = sr.upsample(img1)
img2 = sr.upsample(img2)
img1 = cv2.pyrUp(img1)
img2 = cv2.pyrUp(img2)

# deal with coke image
img_logo = cv2.imread("images/coke_bottle_rotated.png",0)
compare_img_coke = cv2.imread("images/coke_bottle.png")
#print(img1.shape[:2])


kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
kp_logo, des_logo = orb.detectAndCompute(img_logo, None)
kp_compare_coke, des_compare_coke = orb.detectAndCompute(compare_img_coke, None)


imgKp1 = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
imgKp1 = cv2.drawKeypoints(img1, kp1, None)
imgKp2 = cv2.drawKeypoints(img2, kp2, None)
imgKp_logo = cv2.drawKeypoints(img_logo, kp_logo, None)

cv2.imshow('imgKp1', imgKp1)
cv2.imshow('imgKp2', imgKp2)
cv2.imshow('imgKp_logo', imgKp_logo)

# match method 2
bf = cv2.BFMatcher()
try:
    matches1 = bf.knnMatch(des1, des2, k=2)
except:
    print("matches1 failed")
matches_coke = bf.knnMatch(des_logo,des_compare_coke,k=2)

print(matches1)
good1 = []
for m,n in matches1:
    if m.distance < 0.75*n.distance:
        good1.append([m])

good4 = []
for m,n in matches_coke:
    if m.distance < 0.75*n.distance:
        good4.append([m])


img_c1 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good1, None, flags=2)
img_c_coke = cv2.drawMatchesKnn(img_logo, kp_logo, compare_img_coke, kp_compare_coke, good4, None, flags=2)
#print(len(good4))
cv2.imshow('result_img', img_c1)
cv2.imshow('result', img_c_coke)
cv2.waitKey(0)

