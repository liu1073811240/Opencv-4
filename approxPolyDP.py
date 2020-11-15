import cv2
import numpy as np

# 轮廓近似：approxPolyDP()   它主要功能是把一个连续光滑曲线折线化，对图像轮廓点进行多边形拟合。
img = cv2.imread("./images/22.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 第一个参数是寻找轮廓的图像。 cv2.RETR_TREE建立一个等级树结构的轮廓。
# cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 要求传入二值图
# contours返回值首先返回一个list，list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示.
# hierarchy 这是一个ndarray，其中的元素个数和轮廓个数相同.

print(type(contours))  # <class 'list'>
print(type(contours[0]))  # <class 'numpy.ndarray'>
print(len(contours))  # 1

print(type(hierarchy))  # <class 'numpy.ndarray'>
print(hierarchy.ndim)  # 3
print(hierarchy[0].ndim)  # 2

# 轮廓近似：
# approxPolyDP(curve, epsilon, closed, approxCurve=None)
# epsilon指定逼近精度的参数。这是原始曲线与其近似值之间的最大距离。
epsilon = 20  # 精确度，越小越精确
approx = cv2.approxPolyDP(contours[0], epsilon, True)
print(np.shape(approx))  # (5, 1, 2)
print(approx)

# 绘制轮廓：直接对原图进行操作
cv2.drawContours(img, [approx], -1, (0, 0, 255), 3)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

