# conda activate py38
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Super Resolution
path = "ESPCN/ESPCN_x4.pb"
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(path)
sr.setModel("espcn",4)


orb = cv2.ORB_create(nfeatures=1000)
prev_path = "previous_img/"
cur_path = "current_img/"
prev_img = []
cur_img = []
pre_img_list = os.listdir(prev_path)
print(pre_img_list)
cur_img_list = os.listdir(cur_path)
print(cur_img_list)
#print('Total Classes Detected', len(pre_img_list))
class_names = []

for cl in pre_img_list:
    img = cv2.imread(f'{prev_path}/{cl}',0)
    img = sr.upsample(img) # Super Resolution
    img = cv2.pyrUp(img)
    prev_img.append(img)
    class_names.append(os.path.splitext(cl)[0])

print(class_names)
for cl in cur_img_list:
    img = cv2.imread(f'{cur_path}/{cl}',0)
    img = sr.upsample(img) # Super Resolution
    img = cv2.pyrUp(img)
    cur_img.append(img)

def findDes(images):
    desList=[]
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

prev_desList = findDes(prev_img)
#print(len(prev_desList))
#cur_desList = findDes(cur_img)
#print(len(cur_desList))

def findID(img, prev_desList, thres):
    cur_kpn, cur_des = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for prev_des in prev_desList:
            matches = bf.knnMatch(prev_des, cur_des, k=2)
            good = []
            for m, n in matches:
                if m.distance <0.75 *n.distance:
                    good.append([m])
            #print(len(good))
            matchList.append(len(good))
        #print(matchList)
    except:
        print("pass")
        pass

    if len(matchList)!=0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))

    return finalVal

img_index = 2
#print("bring current frame object" + cur_img_list[img_index] + "to find objcet on the previous frame")
rec_id = findID(cur_img[img_index], prev_desList, 1)
cv2.imshow('img', cur_img[img_index])
if rec_id != -1:
    regc = class_names[rec_id]
    cv2.putText(cur_img[img_index], class_names[rec_id], (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    #print(class_names[rec_id])
    #cv2.imshow('img', cur_img[img_index])
    cv2.imshow('img', prev_img[class_names.index(regc)])
    cv2.waitKey(0)
    print(pre_img_list)
    print("bring current frame object:" + cur_img_list[img_index] + " to find objcet on the previous frame")
    print(class_names[rec_id])
else:
    print("bring current frame object:" + cur_img_list[img_index] + " to find objcet on the previous frame")
    print("couldn't find this object on the previous frame")
    
