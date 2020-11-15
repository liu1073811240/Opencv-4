import cv2

# 读取图片
raw_image = cv2.imread("./images/27.jpg")

# 高斯模糊，将图片平滑化，去掉干扰的噪声
image = cv2.GaussianBlur(raw_image, (3, 3), 0)

# 图片灰度化
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Sobel算子（X方向）
Sobel_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
Sobel_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(Sobel_x)  # 转回uint8
absY = cv2.convertScaleAbs(Sobel_y)
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
image = dst

cv2.imshow("image", image)

# 二值化：图像的二值化，就是将图像上的像素点的灰度值设置为0或255，图像中显示出明显的有黑和白
ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
cv2.imshow("image1", image)

# 闭操作：闭操作可以将目标区域连成一个整体，便于后续轮廓的提取
kernel_X = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_X)
cv2.imshow("image2", image)

# 膨胀腐蚀（形态学处理）
kernel_X = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
kernel_Y = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))

image = cv2.dilate(image, kernel_X)  # 在x轴上膨胀，填充车牌漏洞，会在y轴上连续
image = cv2.erode(image, kernel_X)  # 在x轴上腐蚀，变回原状
image = cv2.erode(image, kernel_Y)  # 在y轴上腐蚀，切断x轴上数据
image = cv2.dilate(image, kernel_Y)  # 在y轴上膨胀，返回原型
image = cv2.medianBlur(image, 15)  # 平滑处理，中值滤波
cv2.imshow("image3", image)

# 查找轮廓
contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for item in contours:
    rect = cv2.boundingRect(item)  # 根据轮廓点来找到矩形框
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    if w > (h * 2):
        # 裁剪区域图片
        chepai = raw_image[y:y+h, x:x+w]
        cv2.imshow("chepai"+str(x), chepai)

# 绘制轮廓
image = cv2.drawContours(raw_image, contours, -1, (0, 0, 255), 3)
cv2.imshow("image4", image)

cv2.waitKey(0)
cv2.destroyAllWindows()










