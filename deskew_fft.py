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
                peak_top_bottom:bool=True,
                synth:Optional[bool]=None
                ) -> None:
        """
        Args:
            --- Synth used if image rotated manually where edges of image are black.
                Black pixel grids affect noise in FFT. 
                If you inference set synth to False
            --- peak_top_bottom is boolean parameter. If set False choosing peaks
                of thresholded Fourier image use all (left,right,top,bottom) peaks 
        """
        self.image_path = img_path
        self.resize_ration = resize_ratio
        self.visualize = visualize
        self.peak_top_bottom = peak_top_bottom
        self.synth = synth
        

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
                fft:np.ndarray,
                site:str='Left'
                ) -> Tuple:

        if site == 'top':
            for x in range(fft.shape[0]):
                for y in range(fft.shape[1]):
                    if (fft[x,y]) != 0:
                        return x,y

        if site == 'bottom':
            for x in range(fft.shape[0]-1,0,-1):
                for y in range(fft.shape[1]):
                    if (fft[x,y]) != 0:
                        return x,y
                
        if site == 'left':
            for y in range(fft.shape[1]):
                for x in range(fft.shape[0]):
                    if (fft[x,y]) != 0:
                        return x,y
        
        if site == 'right':
            for x in range(fft.shape[0]):
                for y in range(fft.shape[1]-1,0,-1):
                    if (fft[x,y]) != 0:
                        return x,y

    def fourierTransform(self,
			image:np.ndarray,
			) -> np.ndarray:

        if self.synth:
            if image.max() > 1:
                image[image==0] = 255
            else:
                image[image==0] = 1

        f_img = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_img)
        f_shift = 20*np.log(np.abs(f_shift)) #20* - for stability
        f_shift = (f_shift - f_shift.min())*(255/(f_shift.max()-f_shift.min()))
        f_shift = f_shift.astype(np.uint8)
        return f_shift

    def calc_eucl(self,
                  lr:tuple,
                  tb:tuple):
        lr_dist = np.sqrt((lr[2]-lr[0])**2 + (lr[3]-lr[1])**2)
        tb_dist = np.sqrt((tb[2]-tb[0])**2 + (tb[3]-tb[1])**2)

        return tb if tb_dist > lr_dist else lr

    def deskew(self,
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
                  x1:float,
                  y1:float,
                  x2:float,
                  y2:float
                  ) -> float:
        slope = (y2-y1)/(x2-x1) 
        """
        if we have line with formula y=kx+b then tg(alfa)=k
        k-is a slope of a line
        https://lms2.sseu.ru/courses/eresmat/course1/razd9z1/par9_6z1.htm
        """
        return math.degrees(math.atan(-slope))


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

        image_fft = self.fourierTransform(image_rotated)
        if self.visualize:
            plt.imshow(image_fft)
            plt.show()
        angle = self.deskew(image_fft)
        image = self.rotateImage(image,angle,True)
        if self.visualize:
            plt.imshow(image)
            plt.show()
        return angle, image



def test_rotate_class():
    path = 'test_image_1.jpg'
    for angle in range(-60,70,10):
        print(angle)
        rotated_image = rotate(io.imread(path),angle)
        rotate_doc = RotateDoc(
                               img_path=path,
                               visualize=False,
                               resize_ratio=2,
                               peak_top_bottom=True,
                               synth=True
                               )
        ag, image_rotate = rotate_doc.deskewImage(angle=angle)
        print(ag)
        rotated_image = rotate(rotated_image,ag)
        print('-'*100)
        


if __name__ == '__main__':        
    test_rotate_class()
