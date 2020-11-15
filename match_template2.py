import cv2
import numpy as np

# 多对象匹配，原图中有多个模版匹配
img = cv2.imread("./images/19.jpg")
template = cv2.imread("./images/20.jpg")

h, w, c = template.shape
print(template.shape)  # (35, 35, 3)

# 匹配模版，得到匹配灰度图
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)  # 最大值为匹配区域
print(res.shape)  # h, w  符合条件的有326*446个像素
# print(res)  # 像素值

# 当匹配像素值>=0.95,我们认为是匹配的。
locs = np.where(res >= 0.95)  # 返回匹配程度大于0.95的数组。
print(locs)  # (array([230, 230, 230, 230, 230], dtype=int64), array([125, 163, 200, 310, 384], dtype=int64))
print(*locs[::-1])  # w,h -> x, y     [125 163 200 310 384] [230 230 230 230 230]

for pt in zip(*locs[::-1]):
    print(pt[0], pt[1])

    # 最后，将所有匹配的区域标记出来
    cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), color=(0, 0, 255), thickness=1)

cv2.imshow("img", img)
cv2.imshow("template", template)
cv2.waitKey(0)
cv2.destroyAllWindows()


