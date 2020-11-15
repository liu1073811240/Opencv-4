'''
1.轮廓检测算法公式检测出轮廓：使用参数作为坐标系
2.投射到Hough空间进行形状检测
  1）直线检测
    lines = cv2.HoughLines(image, rho, theta, threshold)

    参数:
    image: 单通道的二进制图像。
    rho: (ρ，θ)中ρ的精度。
    theta: (ρ，θ)中θ的精度。
    threshold: 阈值，(ρ，θ)对应的最低投票数。>=threshold被检测为一条线。
  2）圆检测
    circles = cv2.HoughCircles(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None)

    参数:
    method：定义检测图像中圆的方法。目前唯一实现的方法是HOUGH_GRADIENT。
    dp：累加器分辨率与图像分辨率的反比。
      dp=1，则累加器与输入图像具有相同的分辨率；dp=2，累加器有一半的宽度和高度。
    minDist：该参数是让算法能明显区分的两个不同圆之间的最小距离。
    param1 ：用于Canny的边缘阈值上限，下限被置为上限的一半。
    param2：HOUGH_GRADIENT方法的累加器阈值(最低投票数)。阈值越小，检测到的圈子越多。
    minRadius ：最小圆半径。
    maxRadius：最大圆半径。
'''
# 1.直线检测
import cv2
import numpy as np

img = cv2.imread("./images/24.jpg")
img = cv2.GaussianBlur(img, (5, 5), 50)

# 轮廓检测算法检测出轮廓
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 150)  # 提取轮廓边缘信息
# cv2.imshow("edges", edges)

# 2.投射到Hough空间进行形状检测
# 任何一条线都可以用(ρ，θ)这两个术语表示。
# 1）先定义一个累加器，(ρ，θ)对应直线，ρ和θ都分别依次增大(根据精度)，计算每对(ρ，θ)的投票数。
#    其中，ρ以像素为单位，θ以弧度为单位。rho和theta是ρ和θ的精度。
# 2）然后，根据threshold(阈值，最低投票数)来判断是否归为一条直线
lines = cv2.HoughLines(edges, 1, np.pi/30, 100)  # 隔一度的采样，距离

for line in lines:
    rho, theta = line[0]
    print(line[0])
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = rho * a
    y0 = rho * b
    # k1*k2=-1 ==> k2=-1/k1
    # k1 = tan(θ) ==> k2 = -1/tan(θ)=-cot(θ)
    x1 = int(x0 + 1000 * (b))  # 直线起点横坐标
    y1= int(y0 + 1000 * (a))  # 直线起点在纵坐标
    x2 = int(x0 - 1000 * (b))  # 直线终点横坐标
    y2 = int(y0 - 1000 * (a))  # 直线终点纵坐标
    # 画线
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imshow("img", img)
cv2.imshow("gray", gray)
cv2.imshow("edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()










