import cv2
import numpy as np

'''轮廓查找与绘制： findContours(), drawContours()'''
# img = cv2.imread("./images/21.jpg")
img = cv2.imread("./images/22.jpg")
cv2.imshow("img", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh", thresh)

# 查找轮廓:包括的canny算法
# findContours(image, mode, method, contours=None, hierarchy=None, offset=None)
# image：输入图像(二值化图像)
# mode：轮廓检索方式
# method：轮廓近似方法
'''
轮廓检索方式:
cv2.RETR_EXTERNAL	只检测外轮廓
cv2.RETR_LIST	检测的轮廓不建立等级关系
cv2.RETR_CCOMP	建立两个等级的轮廓，上面一层为外边界，里面一层为内孔的边界信息
cv2.RETR_TREE	建立一个等级树结构的轮廓，包含关系
'''
'''
轮廓近似方法:
cv2.CHAIN_APPROX_NONE	存储所有边界点
cv2.CHAIN_APPROX_SIMPLE	压缩垂直、水平、对角方向，只保留端点
'''

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours[0]))  # 点的数量396
print(np.shape(contours))  # (1, 396, 1, 2)
print(hierarchy)  # 层次树 [[[-1 -1 -1 -1]]]
cv2.imshow("thresh2", thresh)

# 绘制轮廓：直接对原图进行操作
# drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
# contourIdx 轮廓的索引（当设置为-1时，绘制所有轮廓）

img_contour = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.imshow("img contour", img_contour)

cv2.waitKey(0)
cv2.destroyAllWindows()



