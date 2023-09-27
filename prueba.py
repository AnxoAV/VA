import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt

def adjustIntensity(inImage,inRange=[],outRange=[0,1]):
   minImage = np.min(inImage)
   maxImage = np.max(inImage)

   height,width = inImage.shape[:2] 

   outImage = np.zeros((height,width))

   print(height)
   print(width)

   if not inRange:
    min = minImage
    max = maxImage
   else:
    min = inRange[0]
    max = inRange[1]

   for i in range(height):
    for j in range(width):
        outImage[i,j] = min + ((max - min) * ((inImage[i,j] - minImage)/(maxImage-minImage)))
   
   return outImage 

def main():
    inImage = cv.imread('plantaosucra.jpg',cv.IMREAD_GRAYSCALE)

    hist = cv.calcHist([inImage],[0],None,[256],[0,256])

    plt.plot(hist)
    plt.title('Histograma normal')

    outImage = adjustIntensity(inImage,[0,255],outRange=[0,1])

    # hist2 = cv.calcHist([outImage],[0],None,[256],[0,256])

    # plt.plot(hist2)
    # plt.title('Histograma cambios')
    # plt.show()

    cv.imwrite('imagen_salida.png',outImage)

if __name__ == "__main__":
    main()