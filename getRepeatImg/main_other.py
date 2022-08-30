import numpy
import cv2
from PIL import Image
import time

startTime = time.time()
img = Image.open('./img/test.jpg')

splitW = img.width / 3

crop = img.crop(((img.width - 220) / 2, 200, (img.width - 220) / 2 + 220, 200 + 220))
crop.save('./img/test-part.jpg')


# opencv模板匹配----多目标匹配
# 读取目标图片
target = cv2.imread("test.jpg")
# 读取模板图片
template = cv2.imread("22.jpg")

# 获得模板图片的高宽尺寸
theight, twidth = template.shape[:2]

# 执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
result = cv2.matchTemplate(target, template, cv2.TM_SQDIFF_NORMED)

# 寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# 绘制矩形边框，将匹配区域标注出来
cv2.rectangle(target, min_loc, (min_loc[0]+twidth, min_loc[1]+theight), (0, 0, 225), 2)

# 匹配值转换为字符串
# 对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法min_val越趋近与0匹配度越好，匹配位置取min_loc
# 对于其他方法max_val越趋近于1匹配度越好，匹配位置取max_loc
strmin_val = str(min_val)

# 初始化位置参数
temp_loc = min_loc
other_loc = min_loc
numOfloc = 1

# 第一次筛选----规定匹配阈值，将满足阈值的从result中提取出来
# 对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法设置匹配阈值为0.01
threshold = 0.01
loc = numpy.where(result < threshold)

for other_loc in zip(*loc[::-1]):
    # 第二次筛选----将位置偏移小于5个像素的结果舍去
    if (temp_loc[0]+5 < other_loc[0]) or (temp_loc[1]+5 < other_loc[1]):
        numOfloc = numOfloc + 1
        temp_loc = other_loc
        cv2.rectangle(target, other_loc, (other_loc[0]+twidth, other_loc[1]+theight), (0, 0, 225), 2)

endTime = time.time()
useTime = endTime - startTime
print(endTime - startTime, numOfloc)

cv2.imshow("target", target)
cv2.waitKey()
cv2.destroyAllWindows()
