import cv2
import numpy as np
'模板匹配'
'''
1）模板匹配，得到匹配灰度图
    res = cv2.matchTemplate(image, templ, method, result=None, mask=None)

    参数:
    image: 输入图像
    templ: 模板图像
    method: 模板匹配方法，包括：
    - CV_TM_SQDIFF 平方差匹配法：该方法采用平方差来进行匹配；最好的匹配值为0；匹配越差，匹配值越大。
    - CV_TM_SQDIFF_NORMED 相关匹配法：该方法采用乘法操作；数值越大表明匹配程度越好。
    - CV_TM_CCORR 相关系数匹配法：1表示完美的匹配；-1表示最差的匹配。
    - CV_TM_CCORR_NORMED 归一化平方差匹配法
    - CV_TM_CCOEFF 归一化相关匹配法
    - CV_TM_CCOEFF_NORMED 归一化相关系数匹配法
2）获取最小和最大像素值及它们的位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
3）最后，将匹配的区域标记出来
    cv2.rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None)
'''

# 1.单对象匹配：原图中仅有一个与模版匹配
img = cv2.imread("./images/16.jpg")
template = cv2.imread("./images/17.jpg")

h, w, c = template.shape
print(img.shape)  # (342, 548, 3)
print(template.shape)  # (48, 36, 3)

# 1.匹配模版，得到匹配灰度图
# res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)  # 匹配方法：最大值是最匹配区域
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)  # 归一化为[-1, 1], 1表示100%匹配
# res = cv2.matchTemplate(img, template, cv2.TM_CCORR)  # 最大值是最匹配区域
# res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)  # 归一化[0, 1], 1表示100%匹配
# res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)  # 最小值是最匹配区域
# res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)  # 归一化[0, 1], 0表示100%匹配

print(res.shape)

# 2.获取最小和最大像素值值及它们的位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print(min_val)  # 最小值为：-0.6616084575653076
print(max_val)  # 最大值为：0.996796727180481
print(min_loc)  # 最小值索引：(55, 215)
print(max_loc)  # 最大值索引：(223, 85)

# 3.最后，将匹配的区域标记出来
# 匹配类型是TM_CCOEFF、TM_CCOEFF_NORMED、TM_CCORR、TM_CCORR_NORMED时，最大值是最匹配区域
cv2.rectangle(img, (max_loc[0], max_loc[1]), (max_loc[0]+w, max_loc[1]+h), color=(0, 0, 255), thickness=2)
# 匹配类型是TM_SQDIFF、TM_SQDIFF_NORMED时，最小值是最匹配区域
# cv2.rectangle(img, (min_loc[0], min_loc[1]), (max_loc[0]+w, max_loc[1]+h), color=(0, 0, 255), thickness=2)

cv2.imshow("img", img)
cv2.imshow("template", template)
cv2.waitKey(0)
cv2.destroyAllWindows()






