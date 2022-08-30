import numpy
import cv2
from PIL import Image
import time

startTime = time.time()
img = Image.open('./img/test.jpg')

crop = img.crop(((img.width - 220) / 2, 200, (img.width - 220) / 2 + 220, 200 + 220))
crop.save('./img/test-part.jpg')

# 是否灰度处理
isGray = False

if isGray:
    target = cv2.imread("./img/test.jpg", 0)
    template = cv2.imread("./img/test-part.jpg", 0)
else:
    target = cv2.imread("./img/test.jpg")
    template = cv2.imread("./img/test-part.jpg")

# 获取宽高     
w, h = template.shape[:2]

#执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)

# 匹配阈值
threshold = 0.95

loc = numpy.where(result >= threshold)
total = []
for i in zip(*loc[::-1]):
    total.append(i)

total.sort()

# 偏移量
dx = (1 - threshold) * 200
# 计数
numOfloc = 1

# 绘制矩形边框，将匹配区域标注出来
cv2.rectangle(target, total[0], (total[0][0] + w, total[0][1] + h), [0, 0, 250])
for index in range(0, len(total)):
    if index + 1 == len(total):
        break

    if total[index + 1][0] - total[index][0] > dx:
        cv2.rectangle(target, total[index + 1], (total[index + 1][0] + w, total[index + 1][1] + h), [0, 0, 250])
        numOfloc += 1

endTime = time.time()
useTime = endTime - startTime
print(useTime, numOfloc)

cv2.imshow("target", target)
cv2.waitKey(0)
cv2.destroyAllWindows()
