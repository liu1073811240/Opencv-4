import cv2
import numpy as np

# 轮廓性质
img = cv2.imread("./images/23.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 查找轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 边界矩形
x, y, w, h = cv2.boundingRect(contours[0])
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# 最小面积矩形
rect = cv2.minAreaRect(contours[0])
box = cv2.boxPoints(rect)
box = np.int32(box)
cv2.drawContours(img, [box], -1, (0, 255, 0), 2)

# 最小外接圆
(x, y), radius = cv2.minEnclosingCircle(contours[0])
cv2.circle(img, (int(x), int(y)), int(radius), (255, 0, 0), 2)

# 绘制轮廓
cv2.drawContours(img, contours, -1, (255, 255, 0), 2)

# 1.边界矩形的宽高比
aspect_ratio = float(w) / h
print("边界矩形的宽高比：", aspect_ratio)

# 2.轮廓面积与边界矩形面积之比
area = cv2.contourArea(contours[0])
rect_area = w*h
extent = float(area) / rect_area
print("轮廓面积与边界矩形面积之比：", extent)

# 3.轮廓面积和凸包面积之比
hull = cv2.convexHull(contours[0])  # 凸包和凸性检测
area = cv2.contourArea(contours[0])
hull_area = cv2.contourArea(hull)
solidity = float(area) / hull_area
print("轮廓面积和凸包面积之比:", solidity)

# 4.与轮廓面积相等的圆的直径
area = cv2.contourArea(contours[0])
equi_diameter = np.sqrt(4*area / np.pi)
print("与轮廓面积相等的圆的直径：", equi_diameter)

# 5.对象的方向
ellipse = cv2.fitEllipse(contours[0])
print(ellipse)
print("对象的方向angle:", ellipse[2])

# 绘制一个圆心在（150,124）、长轴78、短轴261、线宽为2的白色椭圆
# cv2.ellipse(img, ellipse, (0, 255, 255), 2)
cv2.ellipse(img, (150, 124), (78, 261), 138, 0, 300, (0, 0, 255), thickness=2)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


