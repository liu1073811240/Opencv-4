import cv2
import matplotlib.pyplot as plt

# 1.将图片转化为灰度图
img = cv2.imread("./images/18.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对于对比度比较暗的图片，可进行高亮处理
abs = cv2.convertScaleAbs(gray, alpha=6, beta=0)

# 形态学操作（去除中间的黑色噪点）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
close = cv2.morphologyEx(abs, cv2.MORPH_CLOSE, kernel)

# 2.高斯平滑
gaussian = cv2.GaussianBlur(close, (5, 5), 0)

# 3.Canny算法
canny = cv2.Canny(gaussian, 50, 150)  # 曲线在50-150之间且大于150，或大于150保留，否则舍弃。

titles = ['img', 'gray', 'abs', 'close', 'gaussian', 'canny']
images = [img, gray, abs, close, gaussian, canny]
# plt.figure(figsize=(10, 10))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()



