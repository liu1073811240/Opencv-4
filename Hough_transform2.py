import cv2
import numpy as np

# 圆检测
img = cv2.imread("./images/25.jpg")

# 1.轮廓检测算法检测出轮廓
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 100)

cv2.imshow("edges", edges)

# 投影到Hough空间进行形状检测
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,  # 隔1度的采样、表示两个圆之间 圆心的最小距离
                           param1=90, param2=20, minRadius=20, maxRadius=300)  #双边阈值（断裂），最小最大半径

'''
cv2.HoughCircles(image, method, dp, minDist, circles, param1, param2, minRadius, maxRadius)

image为输入图像，需要灰度图

method为检测方法,常用CV_HOUGH_GRADIENT

dp为检测内侧圆心的累加器图像的分辨率于输入图像之比的倒数，如dp=1，累加器和输入图像具有相同的分辨率，如果dp=2，累计器便有输入图像一半那么大的宽度和高度

minDist表示两个圆之间圆心的最小距离

param1有默认值100，它是method设置的检测方法的对应的参数，对当前唯一的方法霍夫梯度法cv2.HOUGH_GRADIENT，它表示传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半

param2有默认值100，它是method设置的检测方法的对应的参数，对当前唯一的方法霍夫梯度法cv2.HOUGH_GRADIENT，它表示在检测阶段圆心的累加器阈值，它越小，就越可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了

minRadius有默认值0，圆半径的最小值

maxRadius有默认值0，圆半径的最大值

'''

print(np.uint16(np.around(circles[0, :])))
# 画圆
if not circles is None:
    circle = np.uint16(np.around(circles))
    for i in circle[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

cv2.imshow("gray", gray)
cv2.imshow("edges", edges)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()







