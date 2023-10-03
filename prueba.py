import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt


def dibujarHist(inHist,outHist):
    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.title('Histograma')
    plt.plot(inHist)
    plt.xlim([0,256])
    plt.xlabel('Valor del píxel')
    plt.ylabel('Frecuencia')

    plt.subplot(1,2,2)
    plt.title('Histograma 2')
    plt.plot(outHist)
    plt.xlim([0,256])
    plt.xlabel('Valor del píxel')
    plt.ylabel('Frecuencia')

    plt.tight_layout()
    plt.show()



#Alteración del rango dinámico
def adjustIntensity(inImage,inRange=[],outRange=[0,1]):
   minImage = np.min(inImage)
   maxImage = np.max(inImage)

   height,width = inImage.shape[:2] 

   outImage = np.zeros((height,width))

   if not inRange:
    min = minImage
    max = maxImage
   else:
    min = inRange[0]
    max = inRange[1]

   for i in range(height):
    for j in range(width):
        outImage[i,j] = outRange[0] + (((outRange[1] - outRange[0]) * (inImage[i,j] - min))/(max-min))
   
   return outImage 

#Ecualizacuón de histograma
def equalizeIntensity(inImage,nBins=256):
    height,width = inImage.shape[:2]

    hist = cv.calcHist([inImage],[0],None,[nBins],[0,256])
    hist_acum = np.cumsum(hist)

    t = ((hist_acum/(height*width))*255)

    outImage = t[inImage]

    return outImage

def main():
    inImage = cv.imread('plantaoscura.jpg',cv.IMREAD_GRAYSCALE)

    inImageNorm = inImage/255.0

    hist = cv.calcHist([np.uint8(inImageNorm * 255)],[0],None,[256],[0,256])

    outImage = adjustIntensity(inImageNorm,[],[0,1])
    outImage2 = equalizeIntensity(inImage)

    histdinam = cv.calcHist([np.uint8(outImage * 255)],[0],None,[256],[0,256])
    histequal = cv.calcHist([np.uint8(outImage2 * 255)],[0],None,[256],[0,256])

    dibujarHist(hist,histequal)

    cv.imwrite('imagen_rdinamico.jpg',np.uint8(outImage * 255))
    cv.imwrite('imagen_ecualizada.jpg',outImage2)

if __name__ == "__main__":
    main()