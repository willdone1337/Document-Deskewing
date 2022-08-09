from lib2to3.pgen2.token import OP
from unittest import skip
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from typing import Optional,Tuple,Union
from collections import Counter
from skimage.transform import rotate
from skimage.draw import line
import skimage.io as io



class RotateDoc:
    def __init__(self,
                img_path:str,
                resize_ratio:Union[int,float]=2,
                visualize:Optional[bool]=None,
                peak_top_bottom:bool=True
                ) -> None:
        self.image_path = img_path
        self.resize_ration = resize_ratio
        self.visualize = visualize
        self.peak_top_bottom = peak_top_bottom

    def blurImage(self,
                 image:np.ndarray,
                 iter_:int=3
                 ) -> np.ndarray:
        kernel = np.ones((3,3),dtype=np.uint8)
        image = cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel,iterations=iter_) 
        image = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel,iterations=(iter_ - 1  if iter_ > 2 else 1)) 
        image = cv2.morphologyEx(image,cv2.MORPH_ERODE,kernel,iterations=iter_-1)
        image = cv2.morphologyEx(image,cv2.MORPH_DILATE,kernel,iterations=iter_-1) 
        
        return image
    

    def rotateImage(self,
					image:np.ndarray,
					angle:int,
                    skimg:bool=True
					) -> np.ndarray:
        if skimg:
            return rotate(image,angle)
        
        row,col = image.shape[:2]
        center=tuple(np.array([row,col])/2)
        rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
        new_image = cv2.warpAffine(image, rot_mat, (col,row))
        return new_image


    def readImage(self,
                  path:str,
				 ) -> np.ndarray:
        image = cv2.imread(path)
        h, w =image.shape[:2]
        image = cv2.resize(image,(int(h/self.resize_ration),int(w/self.resize_ration)))
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        return image


    def get_max(self,
                test,
                site:str='Left'
                ) -> Tuple:

        if site == 'top':
            for x in range(test.shape[0]):
                for y in range(test.shape[1]):
                    if (test[x,y]) >= 1:
                        return x,y

        if site == 'bottom':
            for x in range(test.shape[0]-1,0,-1):
                for y in range(test.shape[1]):
                    if (test[x,y]) >= 1:
                        return x,y
                
        if site == 'left':
            for y in range(test.shape[1]):
                for x in range(test.shape[0]):
                    if (test[x,y]) >= 1:
                        return x,y
        
        if site == 'right':
            for x in range(test.shape[0]):
                for y in range(test.shape[1]-1,0,-1):
                    if (test[x,y]) >= 1:
                        return x,y

    def fourierTransform(self,
						image:np.ndarray,
                        syth:Optional[bool]=None
						) -> np.ndarray:
        if syth:# convert black pixels to 255 for Fourier stablity
            if image.max() > 1:
                image[image==0] = 255
            else:
                image[image==0] = 1

        f_img = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_img)
        f_shift = 20*np.log(np.abs(f_shift)) #20*log for stability i guess
        f_shift = (f_shift - f_shift.min())*(255/(f_shift.max()-f_shift.min()))
        f_shift = f_shift.astype(np.uint8)

        return f_shift

    def calc_eucl(self,
                  lr:tuple,
                  tb:tuple):
        lr_dist = np.sqrt((lr[2]-lr[0])**2 + (lr[3]-lr[1])**2)
        tb_dist = np.sqrt((tb[2]-tb[0])**2 + (tb[3]-tb[1])**2)

        return tb if tb_dist > lr_dist else lr


    def skew(self,
            fourier_image:np.ndarray,
            
                      ) -> float:
        
        fourier_image = cv2.fastNlMeansDenoising(fourier_image, None, 20, 7, 21)
        if self.visualize:
            plt.imshow(fourier_image)
            plt.show()
        fourier_image = np.where(fourier_image<150,0,fourier_image)
        if self.visualize:
            plt.imshow(fourier_image)
            plt.show()

        if self.peak_top_bottom:
            top_x,top_y = self.get_max(fourier_image,'top')
            bottom_x,bottom_y = self.get_max(fourier_image,'bottom')
            x1,y1,x2,y2 = top_x,top_y,bottom_x,bottom_y
        else:
            left_x,left_y = self.get_max(fourier_image,'left')
            right_x,right_y = self.get_max(fourier_image,'right')
            top_x,top_y = self.get_max(fourier_image,'top')
            bottom_x,bottom_y = self.get_max(fourier_image,'bottom')
            x1,y1,x2,y2 = long_dist_pair[0], long_dist_pair[1], long_dist_pair[2], long_dist_pair[3]
            long_dist_pair = self.calc_eucl(
                                    (left_x,left_y,right_x,right_y),
                                    (top_x,top_y,bottom_x,bottom_y)
                                    )

        fourier_image = cv2.cvtColor(fourier_image,cv2.COLOR_GRAY2RGB)
        
        ang = self.get_angel(x1,y1,x2,y2)

        if self.visualize:
            fourier_image_copy = np.zeros_like(fourier_image)
            rr,cc = line(x1,y1,x2,y2)
            fourier_image_copy[rr,cc] = [0,255,0]
            plt.imshow(fourier_image_copy)
            plt.show()
        return ang

    def get_angel(self,
                  x1:int,
                  y1:int,
                  x2:int,
                  y2:int
                  ) -> float:
        lineA = ((x1,y1),(x2,y2))

        slope1 = (y2-y1)/(x2-x1) #self.slope(lineA[0][0], lineA[0][1], lineA[1][0], lineA[1][1])
        slope2 = 0 #self.slope(lineB[0][0], lineB[0][1], lineB[1][0], lineB[1][1])

        return math.degrees(math.atan(-slope1))


    def deskewImage(self,
                    angle:Optional[Union[int,float]]=None
                    ) -> Tuple[Union[float,int],np.ndarray]:
        image = self.readImage(self.image_path)# size is downed by 2
        image_blurred = self.blurImage(image,3)
        if self.visualize:
            plt.imshow(image_blurred)
            plt.show()
        if angle:
            image_rotated = self.rotateImage(image_blurred,angle,True)
            if self.visualize:
                plt.imshow(image_rotated)
                plt.show()
        else:
            image_rotated = image_blurred

        image_fft = self.fourierTransform(image_rotated,syth=True)
        if self.visualize:
            plt.imshow(image_fft)
            plt.show()
        angle = self.skew(image_fft)
        image = self.rotateImage(image,angle,True)
        if self.visualize:
            plt.imshow(image)
            plt.show()
        return angle, image


def test_rotate_class():
    # path = 'i3.jpg'
    path = 'test_image_1.jpg'
    for angle in range(-60,70,10):
        print(angle)
        # if angle == 0:
        #     continue
        rotated_image = rotate(io.imread(path),angle)
        rotate_doc = RotateDoc(img_path=path,visualize=False)
        ag, image_rotate = rotate_doc.deskewImage(angle=angle)
        print(ag)
        rotated_image = rotate(rotated_image,ag)
        print('-'*100)
        
test_rotate_class()


# imgs = ['gray_image.jpg',
# 'gray_image_bluerred.jpg',
# 'after_fft.jpg',
# 'after_denoise.jpg',
# 'after_threshold.jpg','after_drawline.jpg',
# 'after_drawline2.jpg','final.jpg']


# for x in imgs:
#     image = cv2.imread(x)
#     image = cv2.resize(image,(256,384))
#     cv2.imwrite(x,image)
