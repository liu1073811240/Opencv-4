import cv2

# 凸包和凸性检测: convexHull()、isContourConvex()
# 函数 cv2.convexHull() 可以用来检测一个曲线是否具有凸性缺陷，并能纠正缺陷
# 函数 cv2.isContourConvex() 可以用来检测一个曲线是不是凸的。它只能返回 True 或 False。

img = cv2.imread("./images/23.jpg")
# img = cv2.imread("./images/22.jpg")
# img = cv2.imread("./images/21.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 查找轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

hull = cv2.convexHull(contours[0])  # 凸包
print(cv2.isContourConvex(contours[0]), cv2.isContourConvex(hull))
# False True
# 说明轮廓曲线是非凸的，凸包曲线是凸的

cv2.drawContours(img, [hull], -1, (0, 0, 255), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

