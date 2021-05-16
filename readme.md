---
title: 'openCV tracker 自動比對並修復ID程式之特徵比對'
disqus: hackmd
---

openCV tracker 自動比對並修復ID程式之特徵比對
===



[TOC]


## 版本資訊
文件版本：v0.0.1
執行環境：ubuntu 20.04
ORB SIFT HOG 各項soure code 與執行方式請參考 四 五 六 七項

## 一、 文件說明
### (1) 主要功能
擷取微小人物特徵,用於前後幀畫面之人物ID比對,畫面人下舉例 
關於人物特徵比較前後幀ID
previous frame:
![](https://i.imgur.com/Uvlu4RX.jpg)

current frame:
![](https://i.imgur.com/DKyeaHn.jpg)

用下一幀人物比較是上一幀的那個人物
![](https://i.imgur.com/vzxaDg2.jpg)


ex: 此幀 ID 可能不同於上一幀,所以比對上一幀的特徵進而修改ID
![](https://i.imgur.com/aneJyez.png)

找到上一幀人物,即可決定此幀 ID
![](https://i.imgur.com/UBKoEfs.png)


### (2) 人物資訊
驗證影片中的人物資訊如下,列出寬高如下,像素寬高皆再60以下,
其中還包含背景畫面

擷取人物（上一幀）
五名
id_001
![](https://i.imgur.com/UBKoEfs.png)

![](https://i.imgur.com/rMajHAG.png)


id_002
![](https://i.imgur.com/fatooKU.png)

![](https://i.imgur.com/6oRHMq3.png)

id_003
![](https://i.imgur.com/RoOomYz.png)

![](https://i.imgur.com/Sk9dqdE.png)

id_004
![](https://i.imgur.com/MjJkHh0.png)

![](https://i.imgur.com/d1ZrKsf.png)

id_005
![](https://i.imgur.com/rr006Z3.png)

![](https://i.imgur.com/AqugFOB.png)



## 二、 各項人物特徵擷取方式之驗證結果
如下資訊，以下為最好的結果
EDSR(Super Resolution in OpenCV) + Image Pyramid  + SIFT + KNN

特徵擷取方式(Feature Extraction)
```gherkin=
ORB (Oriented FAST and Rotated BRIEF) + KNN ： 
驗證結果：FAILED 取不到任何特幀

EDSR(Super Resolution in OpenCV) + Image Pyramid + ORB + KNN ：
驗證結果： only id_004 failed

SIFT(Scale-invariant feature transform) + KNN ：
驗證結果： Failed

Image Pyramid + SIFT + KNN
驗證結果：OK

EDSR(Super Resolution in OpenCV) + Image Pyramid  + SIFT + KNN：
驗證結果：OK(比上一項目更準確,能擷取更多人物特徵)

EDSR(Super Resolution in OpenCV) + Image Pyramid + HOG + KNN ：
驗證結果：OK,但無法認出不再訓練資料集的人物,
例如驗證不再此5名當中的人物資料,卻會辨識為 5 名中的其中一人
```

## 三、 執行環境安裝 
由於使用 python 來撰寫程式,必須安裝  anaconda 
若是對於 anaconda 安裝不熟悉可參考[此篇文章](https://hackmd.io/@NTUTVOTT/SJMXCwn0P)

以下包含所有的必要套件,先照下列步驟安裝 
#### （1）創建 python 環境 
```gherkin=
conda create -n opencv4.5 python=3.8
```
#### （2）切換至 opencv4.5 套件環境
```gherkin=
conda activate opencv4.5
```
![](https://i.imgur.com/TCszR6s.png)

#### （3）安裝必要套件

關於 opencv版本,若要查詢目前的版本有哪些可使用以下方式 
退出 conda 環境 
```gherkin=
conda deactivate
```
查詢 opencv 版本
```gherkin=
pip install opencv-python== 
```
![](https://i.imgur.com/bW99jzp.png)

切回 opencv4.5 環境並安裝如下套件
```gherkin=
conda activate opencv4.5
pip install opencv-python==4.5.1.48
pip install opencv-contrib-python==4.5.1.48
pip install jupyterlab
pip install matplotlib
pip install sklearn
pip install scikit-image
pip install imutils
```
#### （4）測試 jupyterlab 否可執行 
```gherkin=
jupyter-lab
```
開一檔案輸入如下, 驗證 opencv 版本是否正確
![](https://i.imgur.com/7Lumvy6.png)

![](https://i.imgur.com/9bYQ5U7.png)

## 四、 source code 下載 
關於 ORB 特徵擷取ID 比對 
```gherkin=
git clone 
```

關於 SIFT 特徵擷取ID 比對 
```gherkin=
git clone 
```
關於 HOG 特徵擷取ID 比對 
```gherkin=
git clone 
```

## 五、 ORB 測試步驟與結果驗證
source code 下載完成後,進入資料夾,並切換環境
```gherkin=
conda activate opencv4.5
```

執行 jupyter-lab
```gherkin=
jupyter-lab
```

一步步執行 orb_espcn_find_id.ipynb ,並觀察結果
手動 更改img_index = 0~4 驗證是否辨別成功 
除了 id_004驗證失敗,其他都成功
![](https://i.imgur.com/xllLBSX.png)


目前id_004.png 驗證失敗
![](https://i.imgur.com/pbmr29g.png)


## 六、 SFIT 測試步驟與結果驗證


#### 1. SIFT 安裝來源

**SIFT來源1  （opencv套件內建) - 目前驗證成功**
先以此版本進行驗證,請到2.SFIT 擷取特徵並進行ID比對之驗證 

**SIFT來源2 (下載 SIFT lib 使用) - 目前驗證失敗**

:::warning
不需執行此步驟,以下資訊只是暫時先保留使用 SIFT lib 不容易驗證此功能
:::
下載 SIFT LIB
```
please go to the below webiste to downlod SIFT binary file 
建議使用 vlfeat-0.9.20-bin.tar.gz

this SIFT method created by VLFeat
https://www.vlfeat.org/index.html
```
[SFIT dowload](https://www.vlfeat.org/download/)
 

安裝 SIFT
將解壓縮的 vlfeat-0.9.20-bin.tar.gz 放置到讓專案可以使用的位置
![](https://i.imgur.com/psFePHl.png)
資料如下
![](https://i.imgur.com/WUKVVpC.png)

```
make sure your SIFT binary file path is existed when you use it,for example
$ ~/python_test/vlfeat-0.9.20/bin/glnxa64/sift house.pgm --peak-thresh 5 --edge-thresh 10
or you can add to your ~/.bashrc like below
export PATH = ~/python_test/vlfeat-0.9.20/bin/glnxa64:$PATH
```
測試 SIFT（多數會失敗)
參考路徑（以下為 Ivan Yu 筆電）
~/python_test/textbook/PCV/examples


#### 2. SFIT 擷取特徵並進行ID比對之驗證 

source code 下載完成後,進入資料夾,並切換環境
```gherkin=
conda activate opencv4.5
```

執行 jupyter-lab
```gherkin=
jupyter-lab
```

一步步執行 SIFT_espcn_find_id.ipynb ,並觀察結果
手動 更改img_index = 0~4 驗證是否辨別成功 

![](https://i.imgur.com/Nsj31gj.png)


**其他-1 SFIT 比對特徵點  SFIT_test.ipynb**
::: warning
尚未加上 Super Resolution 
:::
可觀察擷取的特徵點 
![](https://i.imgur.com/iBQ1nBF.png)

也可測試任何圖片,可更改資訊如下
![](https://i.imgur.com/8reEO6v.png)



**其他-2 SFIT 比對所有人物特徵點  SFIT_person.ipynb**
::: warning
已加上 Super Resolution 
:::
![](https://i.imgur.com/3sVwLQ8.png)

## 七、 HOG 測試步驟與結果驗證
source code 下載完成後,進入資料夾,並切換環境
```gherkin=
conda activate opencv4.5
```

執行 jupyter-lab
```gherkin=
jupyter-lab
```

一步步執行 HOG_espcn_find_id.ipynb ,並觀察結果
可得知紅框處因為沒再訓練資料集,所以會找最接近訓練資料集的圖片(人物姿勢)
![](https://i.imgur.com/R7sQEf5.png)


## 八、 參考資料
**ORB 相關：**
[ (Oriented FAST and Rotated BRIEF)](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html)
![](https://i.imgur.com/GdmFKL9.png)

[Feature Detection and Matching + Image Classifier Project | OPENCV PYTHON 2020](https://youtu.be/nnH55-zD38I)

**HOG 相關：**
[Histogram of Oriented Gradients explained using OpenCV](https://learnopencv.com/histogram-of-oriented-gradients/)

[opencv︱HOG描述符介绍+opencv中HOG函数介绍（一）](https://blog.csdn.net/sinat_26917383/article/details/69666371)

[使用方向梯度與KNN模型進行影像識別](https://chtseng.wordpress.com/2017/01/09/%e4%bd%bf%e7%94%a8%e6%96%b9%e5%90%91%e6%a2%af%e5%ba%a6%e8%88%87knn%e6%a8%a1%e5%9e%8b%e9%80%b2%e8%a1%8c%e5%bd%b1%e5%83%8f%e8%ad%98%e5%88%a5-1/)

[python+opencv3.4.0 实现HOG+SVM行人检测](https://blog.csdn.net/qq_33662995/article/details/79356939?utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control)

[OpenCV图像处理-HOG特征和应用](https://zhuanlan.zhihu.com/p/75705284)

https://github.com/preethampaul/HOG

**SIFT 相關**
[A Detailed Guide to the Powerful SIFT Technique for Image Matching (with Python code)
](https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/)

[測試用 SIFT 進行特徵比對(降opencv版本 提供 SIFT )
](https://https://blog.csdn.net/cliukai/article/details/102525486)

[Problem extracting descriptors of FAST opencv3](https://answers.opencv.org/question/61764/problem-extracting-descriptors-of-fast-opencv3/?sort=latest)

[This is an implementation of SIFT (David G. Lowe's scale-invariant feature transform) done entirely in Python ](
https://github.com/rmislam/PythonSIFT/)

[Image Registration: From SIFT to Deep Learning ](https://zhuanlan.zhihu.com/p/82705687)

**Super Resolution:**
[Super Resolution in OpenCV](https://learnopencv.com/super-resolution-in-opencv/?ck_subscriber_id=1048263026)

**opencv match 特徵比對：**
https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
https://docs.opencv.org/master/db/d27/tutorial_py_table_of_contents_feature2d.html

[圖像特徵比對(一)-取得影像的特徵點 介紹各項特徵擷取方式](https://zwindr.blogspot.com/2017/02/python-opencv-matchtemplate.html)

**Homography 相關（尚未研究)**
[Feature Matching + Homography to find Objects](https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html)

[Object tracking using Homography - OpenCV 3.4 with python 3 Tutorial 34](https://www.youtube.com/watch?v=I8tHLZDDHr4)


**其他：**
[Learn OpenCV blog github](https://github.com/spmallick/learnopencv/)

參考書python計算機視覺 github
https://github.com/nico/cvbook
https://github.com/jesolem/PCV
http://programmingcomputervision.com/

[Comparing Deep Neural Networks and Traditional Vision
Algorithms in Mobile Robotics](https://www.cs.swarthmore.edu/~meeden/cs81/f15/papers/Andy.pdf)


###### tags: `study`, `feature`
