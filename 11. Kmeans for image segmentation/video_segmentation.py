
import cv2
import cvzone  # mediapipe
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np

cap = cv2.VideoCapture(0)
wid = 320
hei = 240
cap.set(3,wid) # 设置显示宽度
cap.set(4,hei) # 设置显示高度
segmentor = SelfiSegmentation()
#fpsReader = cvzone.FPS()
bk_image = cv2.imread(".\\images\\background.png")
bk_image = cv2.resize(bk_image,[wid,hei])

while True:
    success,img = cap.read()
    imgOut = segmentor.removeBG(img,(255,0,0))

    # foreground clustering
    vectorized = imgOut.reshape((-1,3))
    vectorized = np.float32(vectorized)
    
    # perform kmeans clustering
    ## 停止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    ## 初始化中心数和尝试的次数
    K = 3
    attempts=4
    _,label,center = cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    label.flatten().shape,center.shape
    res = center[label.flatten()]
    imgOut = res.reshape((imgOut.shape))
    
    
    imgStacked = cvzone.stackImages([img,imgOut],2,1)
    # _,imgStacked = fpsReader.update(imgStacked,color = (0,0,255))
    
    cv2.imshow("Image",imgStacked)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break