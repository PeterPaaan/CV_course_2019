#!/usr/bin/env python
# coding: utf-8

# ![logo](CV_course1_image/OpenCV_Logo.png)
# ### **OpenCV**的全称是_Open Source Computer Vision Library_，是一个跨平台的计算机视觉库。OpenCV是由英特尔公司发起并参与开发，以BSD许可证授权发行，可以在商业和研究领域中免费使用。OpenCV可用于开发实时的图像处理、计算机视觉以及模式识别程序。
# 
# - [OpenCV-Python Tutorials](https://opencv-python-tutroals.readthedocs.io/en/latest/) 
# 
# - [openCV-Python中文教程](https://www.kancloud.cn/aollo/aolloopencv/269602) 
# 
# - [openCV_Gui中文文档](https://ptorch.com/docs/6/opencv-imshow-imwrite) 
# 
# ---
# <u>_@peterpan@Pi-Lab 20190404_</u>

# # Low Level Image Processing
# for image data augmentation

# ---
# ### 对图片进行一些底层操作，可用于图片数据集的数据增强。
# - 读取图片
# - 查看图片
# - 按键事件
# - 获取图像属性（行、列、通道）
# - 通道拆分、合并
# - 图片颜色随机变化
# - 颜色空间变换
# - gamma校正
# - Histogram equalized
# - image crop
# - image rotation
# - similarity transformation
# - Affine Transform
# - perspective transform
# 

# In[45]:


import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


# ### cv2.imread('image_path_to_xxx.jpg/png/etc', flag)
# 第一个参数图像应该在工作目录中，或者应该给出完整的图像路径。
# 
# 第二个参数是一个标志，指定应读取图像的方式。
# 
# cv2.IMREAD_COLOR：加载彩色图像。任何形象的透明度将被忽略。这是默认的标志。
# cv2.IMREAD_GRAYSCALE：以灰度模式加载图像;
# cv2.IMREAD_UNCHANGED：加载包含Alpha通道的图像; 
# 除了上面这三个标志，可以简单地传递整数1，0或-1。

# In[39]:


img_gray = cv2.imread('/Users/peterpan/jupyter_notebook/lenna.png', 0) # 以灰度模式读取 路径下的图像
# plt.imshow() 
# plt为RGB模式，opencv为BGR接口
cv2.imshow('lenna', img_gray) # 在新建窗口中显示 如下图
# cv2.imwrite('lenna_gray.png',img_gray) # 将显示的图片写入当前目录
key = cv2.waitKey()  # 按键事件
if key == 27: # 如果ESC被按下，销毁所有图片窗口
    cv2.destroyAllWindows()


# 灰度图： ![logo](CV_course1_image/lenna_gray.png)
# 
# zoom_in： ![logo](CV_course1_image/zoomin.jpeg)

# In[50]:


# show image matrix of the gray image
print(img_gray)


# In[51]:


# to show image data type
print(img_gray.dtype)
# uint8  无符号整型 0到255整数 因此像素数值均为0-255的整数
# int8   有符号整型 -128到127整数


# In[53]:


# to show gray image shape
print(img_gray.shape)
# 原图为 220x220 pixel的彩色图


# ### 加载彩色图像
# 默认模式

# In[54]:


import cv2
img = cv2.imread('/Users/peterpan/jupyter_notebook/lenna.png')
cv2.imshow('lenna', img)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# ![logo](CV_course1_image/lenna.png)

# In[57]:


# to show color image 
# to show channels
#print(img)
print(img[:, :, 0])  # 打印第一个通道
print('-'*50)
print(img[:, :, 0].shape)  # 显示第一个通道的形状
print('-'*50)
print(img.shape)  # 整个图片的形状
# 行 列 通道 220 * 220 * 3 
# 通道在CNN中也叫图片的 高


# ---
# ### cv2.split()
# 通道拆分(顺序为BGR而不是RGB)：
# ```python
# cv2.split(img)
# (B, G, R) = cv2.split(img)
# ```
# 显示各个分离的通道：
# 
# ```cv2.imshow('Blue', B)```
# 
# imshow()显示分离的三个通道B, G, R均为单通道图像，显示出来也都是灰色
# 而不是B为蓝色，G为绿色，R为红色。
# 
# ### cv2.merge()
# 通道合并
# ```cv2.merge((B, G, R))```
# 
# RGB图片分离之后的B、G、R都是单通道图像，显示之后都是黑白的；
# 
# 而不是分别为蓝、绿、红；如果为蓝、绿、红的话，那么仍然是三通道彩色图
# 
# 只不过蓝对应 B通道不必、GR通道的矩阵元素全为0

# In[62]:


# color split
B, G, R = cv2.split(img)
# zero_channel = np.zeros(B.shape, dtype = "uint8")
cv2.imshow('B', B)
cv2.imshow('G', G)
cv2.imshow('R', R)
# cv2.imshow('B', cv2.merge([B, zero_channel, zero_channel]))
# cv2.imshow('G', cv2.merge([zero_channel, G, zero_channel]))
# cv2.imshow('R', cv2.merge([zero_channel, zero_channel, R]))

key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# 从左往右一次为B, G, R通道; 灰度图中像素值一次增大（R通道最大，可zoom in图片查看）。
# ![logo](CV_course1_image/lenna_rgb.jpeg)
# ![logo](CV_course1_image/lenna_rgb_gray.jpeg)
# 
# ---

# ### 随机改变图像颜色
# 将原RGB格式图像拆分后，随机改变每个通道的像素值，然后合并。

# In[64]:


def random_light_color(img):
    # brightness
    B, G, R = cv2.split(img) # 将彩图img拆分为B、G、R三个单色通道
    
    b_rand = random.randint(-50, 50)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255 # 相加超过255就等于255；B为numpy矩阵的时候可以这样操作，对列表会报错
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)# 防止出现类型错误
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)
        
    g_rand = random.randint(-50, 50 )
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)
        
    r_rand = random.randint(-50, 50 )
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)
        
    img_merge = cv2.merge((B, G, R)) # 将3个单通道图片合为三通道图片(彩色)
    # img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV_BGR)
    return img_merge
        


# In[70]:


# 生成随机颜色的图片并存储下来
img_random_color = random_light_color(img)
cv2.imshow('img_random_color', img_random_color)
#cv2.imwrite('img_random_color1.png', img_random_color)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
    


# 随机改变颜色的三张图片：
# ![logo](CV_course1_image/lenna_random_color.jpeg)

# In[45]:


# gamma correction
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_dark = cv2.imread('/Users/peterpan/jupyter_notebook/dark.jpg')
# cv2.imshow('img_dark', img_dark)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# ### gamma校正
# LUT(src, lut[, dst]) -> dst
# 亮度增加

# In[ ]:


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255) # invGamma越小，像素值越靠近255，图片亮度越高
    table = np.array(table).astype("uint8")
    return cv2.LUT(img_dark, table)

img_brighter = adjust_gamma(img_dark, 4)
# cv2.imshow('img_dark', img_dark)
# cv2.imshow('img_brighter', img_brighter)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
    
    


# ![logo](CV_course1_image/gamma.jpeg)

# In[4]:


from matplotlib import pyplot as plt

def curve(x):
    return x**0.5*255
x = np.arange(256)/255
y = np.zeros(256)

for i, e in enumerate(x):
    y[i] = curve(e)

plt.plot(x, y)
plt.xlabel('index', fontsize = 22)
plt.ylabel('gray_scale_value', fontsize = 22)
plt.title(r'$how \ 1/\gamma\ influence\ pixel\ values$', fontsize=22)
plt.show()


# ###  cv2.equalizeHist(图像通道)
# 将制定通道像素分布均匀化
# 见下图histogram红蓝对比

# In[3]:


# histogram
from matplotlib import pyplot as plt
img_small_brighter = cv2.resize(img_brighter, (int(img_brighter.shape[0]*2), int(img_brighter.shape[1]*0.5)))
plt.hist(img_brighter.flatten(), 256, [0, 256], color = 'r', label='unequ')
img_yuv = cv2.cvtColor(img_small_brighter, cv2.COLOR_BGR2YUV)
# equalize the histogram of the Y channel
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # only for the 1st channel
plt.hist(img_yuv.flatten(), 256, [0, 256], color = 'b', label='equ')
plt.legend()
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) #
# fig = plt.figure(figsize=(8, 12))
# plt.subplot(3, 1, 1)
# plt.imshow(img_yuv)
# plt.title('img_yuv')
# plt.subplot(3, 1, 2)
# plt.imshow(img_small_brighter)
# plt.title('img_small_brighter')
# plt.subplot(3, 1, 3)
# plt.imshow(img_output)
# plt.title('img_output')
cv2.imshow('Color input image', img_small_brighter)
cv2.imshow('Histogram equalized', img_output)
key = cv2.waitKey(0)
if key == 27:
    exit()


# ![logo](CV_course1_image/histogram.jpeg)
# 
# ---

# ### image crop
# 截取图像中的某块区域

# In[27]:


# image crop
import cv2
img_gray = cv2.imread('/Users/peterpan/jupyter_notebook/lenna.png', 0) 
img = cv2.imread('/Users/peterpan/jupyter_notebook/lenna.png')
img_crop1 = img_gray[20:150, 20:150]
img_crop2 = img[50:180, 50:180]
cv2.imshow('img_crop2', img_crop2)
cv2.imshow('img_crop1', img_crop1)

key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# ![logo](CV_course1_image/crop.jpeg)

# ### image rotation
# 用一个旋转操作矩阵乘以原像素矩阵

# In[40]:


# rotation
M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), -45, 1.2)  # center, angle, scale
img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('rotated lenna', img_rotate)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()
    
print(M)


# ![logo](CV_course1_image/rotation.jpeg)

# In[42]:


# set M[0][2] = M[1][2] = 0
print(M)
img_rotate2 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('rotated lenna2', img_rotate2)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()


# ### similarity truansformation
# 相似变换：空间保持正交、平行、线仍是直线

# In[44]:


# scale + rotation + translation = similarity transformation
# 等比伸缩 + 旋转 + 平移 = 相似变换
M = cv2.getRotationMatrix2D((img.shape[1]/3, img.shape[0]/3), 30, 0.75) # center, angle, scale
img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow('rotated lenna', img_rotate)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()
    
print(M)


# ![logo](CV_course1_image/similarity.jpeg)
# 
# ---

# ### Affine Transform
# 仿射变换
# 
# 坐标系仍未直线、平行；但已非正交。

# In[25]:


# Affine Transform
import cv2
img = cv2.imread('/Users/peterpan/jupyter_notebook/lenna.png')
rows, cols, ch = img.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.1, rows * 0.2], [cols * 0.1, rows * 0.5]])
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('affine lenna', dst)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()


# ![logo](CV_course1_image/affine.jpeg)
# 
# ---

# ### perspective transform
# 投影变换
# 
# 坐标系仍为直线，但非正交非平行

# In[35]:


# perspective transform
import random
def random_warp(img, row, col):
    height, width, channels = img.shape

    # warp:
    random_margin = 70
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width-random_margin-1, width-1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width-random_margin-1, width-1)
    y3 = random.randint(height-random_margin-1, height-1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height-random_margin-1, height-1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width-random_margin, width-1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width-random_margin-1, width-1)
    dy3 = random.randint(height-random_margin-1, height-1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height-random_margin-1, height-1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp
                        
M_warp, img_warp = random_warp(img, img.shape[0], img.shape[1])
cv2.imshow('lenna_warp', img_warp)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()     


# ![logo](CV_course1_image/perspective.jpeg)
# 
# ---

# In[ ]:





# In[ ]:


class image_augmentation(object):
    
    
    def img_crop(self，img):
        img_crop = img[0:150, 0:150]
        cv2.imshow('img_crop', img_crop)
        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()
        pass
    
    
    def img_color_shift(self):
        pass
    
    
    def img_rotation(self):
        pass
    
    
    def img_perspective(self):
        pass
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




