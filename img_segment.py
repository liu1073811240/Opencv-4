import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./images/26.jpg")
cv2.imshow("img", img)

# 1.图像二值化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("thresh", thresh)
cv2.imshow("gray", gray)

kernel = np.ones((3, 3), dtype=np.uint8)

# 2.噪声去除
open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imshow("open", open)

# 3.确定背景区域
sure_bg = cv2.dilate(open, kernel, iterations=3)
cv2.imshow("sure_bg", sure_bg)

# 4.寻找前景区域
dist_transform = cv2.distanceTransform(open, 1, 5)  # 距离腐蚀，计算距离
cv2.imshow("dist transform", dist_transform)

# 根据前景像素值来确定阈值大小
ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, cv2.THRESH_BINARY)
cv2.imshow("sure_fg", sure_fg)

# 5.找到未知区域
sure_fg = np.uint8(sure_fg)
unknow = cv2.subtract(sure_bg, sure_fg)  # 背景减去前景，去除背景
cv2.imshow("unknow", unknow)

# 6.类别标记：计算中心
ret, markers = cv2.connectedComponents(sure_fg)  # ret表示前景标记数，markers表示标记的是背景
# 为所有的标记加1， 保证背景是0不是1
markers = markers + 1
markers[unknow == 255] = 0
print(markers)

# 7.分水岭算法
markers = cv2.watershed(img, markers)
img[markers == -1] = (0, 0, 255)
cv2.imshow("img_watershed", img)

cv2.waitKey(0)
cv2.destroyAllWindows()



