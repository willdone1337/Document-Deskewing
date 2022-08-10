# Deskew Scanned Document 
  ### Deskewing document is one of the essemtial preprcessing before OCR and HTR. This repo contain proposed method for document deskewing mainly via Fourier Transform.
## Techniques
---
+ **Fourier Transform**
  
>[Fast Fourier Transform](https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html)

---
+ **Non Local Mean Denoising**

>[Article about NLMD](http://www.ipol.im/pub/art/2011/bcm_nlm/article.pdf)

>[Implementation from scratch](http://dsvision.github.io/an-approach-to-non-local-means-denoising.html)


___

## Requirements
+ OpenCV
+ Scikit-image
+ Numpy
____
## Algorithm steps
+ `Change image to gray format`
+ `Morphological operations`
+ `Fast Fourier Transform`
+ `FNlM Denoising and thresholding of FFT output`
+ `Determines the peaks`
+ `Calculate angle using slope of drawing line`

## P.S.
*This method works well if angle of skew is between -60 and 60 degrees**


___
## Vizualisation of each step
<img src="test_image/gray_image.jpg" width=128px></img>
<img src="test_image/gray_image_bluerred.jpg" width=128px></img>
<img src="test_image/after_fft.jpg" width=128px></img>
<img src="test_image/after_denoise.jpg" width=128px></img>
<img src="test_image/after_threshold.jpg" width=128px></img>
<img src="test_image/after_drawline.jpg" width=128px></img>
<img src="test_image/final.jpg" width=128px></img>

___
## Code usage
```python
from test_doc import RotateDoc
from skimage.transform import rotate
import skimage.io as io
import matplotlib.pyplot as plt

path = 'test_image_1.jpg'
rotate_doc = RotateDoc(
            img_path=path,
            visualize=1,
            resize_ratio=2,
            peak_top_bottom=True,
            synth=False
            )

image = io.imread(path)
ag, image_rotate = rotate_doc.deskewImage()
rotated_image = rotate(image,ag)
if rotate_doc.visualize:
    plt.imshow(rotated_image)
    plt.show()
```


<!-- ![gray](test_image/gray_image.jpg)
![bluer](test_image/gray_image_bluerred.jpg)
![fft](test_image/after_fft.jpg)
![afterNMS](test_image/after_denoise.jpg)
![afterthresh](test_image/after_threshold.jpg)
![afterline1](test_image/after_drawline.jpg)
![final](test_image/final.jpg) -->
___

## Synthetic test
**In this test document manually rotated from -60 to 60 degree and check performance of methdod.**
Rotate -60 degree                       | Deskew 63 degree  
:-----------------------------------:|:-------------------------:
<img src="saved_images/image_-60.jpg" width=128px></img> | <img src="saved_images/image_63_after.jpg" width=128px></img>


Rotate -50 degree                       | Deskew 53 degree  
:-----------------------------------:|:-------------------------:
<img src="saved_images/image_-50.jpg" width=128px></img> | <img src="saved_images/image_53_after.jpg" width=128px></img>


Rotate -40 degree                       | Deskew 46 degree  
:-----------------------------------:|:-------------------------:
<img src="saved_images/image_-40.jpg" width=128px></img> | <img src="saved_images/image_46_after.jpg" width=128px></img>


Rotate -30 degree                       | Deskew 35 degree  
:-----------------------------------:|:-------------------------:
<img src="saved_images/image_-30.jpg" width=128px></img> | <img src="saved_images/image_35_after.jpg" width=128px></img>



Rotate -20 degree                       | Deskew 25 degree  
:-----------------------------------:|:-------------------------:
<img src="saved_images/image_-20.jpg" width=128px></img> | <img src="saved_images/image_25_after.jpg" width=128px></img>


Rotate -10 degree                       | Deskew 12 degree  
:-----------------------------------:|:-------------------------:
<img src="saved_images/image_-10.jpg" width=128px></img> | <img src="saved_images/image_12_after.jpg" width=128px></img>


Rotate 10 degree                       | Deskew -12 degree  
:-----------------------------------:|:-------------------------:
<img src="saved_images/image_10.jpg" width=128px></img> | <img src="saved_images/image_-12_after.jpg" width=128px></img>


Rotate 20 degree                       | Deskew -24 degree  
:-----------------------------------:|:-------------------------:
<img src="saved_images/image_20.jpg" width=128px></img> | <img src="saved_images/image_-24_after.jpg" width=128px></img>

Rotate 30 degree                       | Deskew -35 degree  
:-----------------------------------:|:-------------------------:
<img src="saved_images/image_30.jpg" width=128px></img> | <img src="saved_images/image_-35_after.jpg" width=128px></img>

Rotate 40 degree                       | Deskew -47 degree  
:-----------------------------------:|:-------------------------:
<img src="saved_images/image_40.jpg" width=128px></img> | <img src="saved_images/image_-47_after.jpg" width=128px></img>

Rotate 50 degree                       | Deskew -55 degree  
:-----------------------------------:|:-------------------------:
<img src="saved_images/image_50.jpg" width=128px></img> | <img src="saved_images/image_-55_after.jpg" width=128px></img>

Rotate 60 degree                       | Deskew -65 degree  
:-----------------------------------:|:-------------------------:
<img src="saved_images/image_60.jpg" width=128px></img> | <img src="saved_images/image_-65_after.jpg" width=128px></img>


<!-- 
![alt-text-1](saved_images/image_10.jpg "title-1") ![alt-text-2](saved_images/image_-12_after.jpg "title-2")
![alt-text-1](saved_images/image_20.jpg "title-1") ![alt-text-2](saved_images/image_-24_after.jpg "title-2")
![alt-text-1](saved_images/image_30.jpg "title-1") ![alt-text-2](saved_images/image_-35_after.jpg "title-2")
![alt-text-1](saved_images/image_40.jpg "title-1") ![alt-text-2](saved_images/image_-47_after.jpg "title-2")
![alt-text-1](saved_images/image_50.jpg "title-1") ![alt-text-2](saved_images/image_-55_after.jpg "title-2")
![alt-text-1](saved_images/image_60.jpg "title-1") ![alt-text-2](saved_images/image_-65_after.jpg "title-2") -->
___

---
## Synthetic test code
```python 
from test_doc import RotateDoc
from skimage.transform import rotate
import skimage.io as io
import matplotlib.pyplot as plt

path = 'test_image_1.jpg'
for angle in range(-60,70,10):
    rotated_image = rotate(io.imread(path),angle)
    rotate_doc = RotateDoc(
                            img_path=path,
                            visualize=1,
                            resize_ratio=2,
                            peak_top_bottom=True,
                            synth=True
                            )
    ag = rotate_doc.deskewImage(angle=angle)
    rotated_image = rotate(rotated_image,ag)
    if rotate_doc.visualize:
        plt.imshow(rotated_image)
        plt.show()
    print('-'*100)
```

