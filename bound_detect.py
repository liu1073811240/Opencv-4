import cv2
import numpy as np

# "边界检测: 边界矩形、最小(面积)矩形、最小外接圆以及椭圆拟合、直线拟合"

img = cv2.imread("./images/23.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 边界矩形
x, y, w, h = cv2.boundingRect(contours[0])  # 根据轮廓点来获取边界框的坐标
img_contour = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow("img_contour", img_contour)

# 最小矩形
rect = cv2.minAreaRect(contours[0])  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
print(rect)

box = cv2.boxPoints(rect)   # 获取最小外接矩形的4个顶点坐标
print(box)
print(box.dtype, box.shape)

box = np.int32(box)
print(box.dtype, box.shape)

img_contour1 = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
cv2.imshow("img_contour1", img_contour1)

# 最小外接圆
(x, y),radius = cv2.minEnclosingCircle(contours[0])  # 根据轮廓点找到最小闭合圆的中心点坐标，半径
center = (int(x), int(y))
radius = int(radius)
img_contour3 = cv2.circle(img, center, radius, (255, 0, 0), 2)
cv2.imshow("img_contour3", img_contour3)

# 椭圆拟合
ellipse = cv2.fitEllipse(contours[0])  # 根据轮廓点找到椭圆
print(ellipse)
img_contour4 = cv2.ellipse(img, ellipse, (0, 255, 255), 2)
cv2.imshow("img_contour4", img_contour4)

cv2.waitKey(0)
cv2.destroyAllWindows()


