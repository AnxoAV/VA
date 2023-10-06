import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt
import math


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

#Filtro espacial de suavizado
def filterImage(inImage,kernel):
    P,Q = kernel.shape
    height,width = inImage.shape[:2]

    outImage = np.zeros((height,width))

    for i in range(height):
        for j in range(width):
            tmp = 0
            for k in range(P):
                for l in range(Q):
                    x = i - (P//2) + k
                    y = j - (Q//2) + l

                    if x >= 0 and x < height and y >= 0 and y < width:
                        tmp += inImage[x,y] * kernel[k,l]

            outImage[i,j] = tmp

    return outImage

#Calculo de kernel Gaussiano
def gaussKernel1D(sigma):
    N = int(2 * np.ceil(3 * sigma) + 1)
    kernel = np.zeros(N)
    centro = (N - 1) // 2

    for i in range(N):
        #kernel[i] = np.exp(-(x - centro) ** 2) / (2 * sigma **2)  PREGUNTAR
        kernel[i] = (1.0 / (math.sqrt(2 * math.pi * sigma))) * math.exp(-((i-centro) ** 2)) / (2 * sigma ** 2)

    kernel /= np.sum(kernel)  #Preguntar también

    return kernel

#Filtro de suavizado Gaussiano
def gaussianFilter(inImage,sigma):
    kernel = gaussKernel1D(sigma)
    kernel2D = kernel[np.newaxis,:]     #PREGUNTAR
    tmpImage = filterImage(inImage,kernel2D)
    outImage = filterImage(tmpImage,kernel2D.reshape(-1,1))

    return outImage

#Filtro de medianas bidimensional
def medianFilter(inImage,filterSize):
    height,width = inImage.shape

    centro = filterSize // 2

    outImage = np.zeros((height,width))

    for i in range(height):
        for j in range(width):
            row_ini = max(0,i - centro)
            row_fin = min(height,i + centro + 1)
            col_ini = max(0,j - centro)
            col_fin = min(width, j + centro + 1)

            dimension = inImage[row_ini:row_fin,col_ini:col_fin]

            mediana = np.median(dimension)

            outImage[i,j] = mediana
    
    return outImage

#Test para probar el filtro de medianas
def testMedianFilter():
    inImage = cv.imread('ruidoimpulsional.jpeg',cv.IMREAD_GRAYSCALE)
    inImageNorm = inImage / 255.0
    filterSize = 7

    outImage = medianFilter(inImageNorm,filterSize)

    cv.imwrite('imagen_medianas.jpg',np.uint8(outImage * 255))        


def main():
    inImage = cv.imread('ruidoimpulsional.jpeg',cv.IMREAD_GRAYSCALE)

    inImageNorm = inImage/255.0

    hist = cv.calcHist([np.uint8(inImageNorm * 255)],[0],None,[256],[0,256])

    outImage = adjustIntensity(inImageNorm,[],[0,1])
    outImage2 = equalizeIntensity(inImage)

    kernel = np.array([[1,1,1],
                       [1,1,1],
                       [1,1,1]])/9.0


    outImage3 = filterImage(inImageNorm,kernel)

    histdinam = cv.calcHist([np.uint8(outImage * 255)],[0],None,[256],[0,256])
    histequal = cv.calcHist([np.uint8(outImage2 * 255)],[0],None,[256],[0,256])


    outImage4 = gaussianFilter(inImageNorm,2)

    #dibujarHist(hist,histequal)

    cv.imwrite('imagen_rdinamico.jpg',np.uint8(outImage * 255))
    cv.imwrite('imagen_ecualizada.jpg',outImage2)
    cv.imwrite('imagen_filtrada.jpg',np.uint8(outImage3 * 255))
    cv.imwrite('imagen_Gauss.jpg',np.uint8(outImage4 * 255))
    
    testMedianFilter()

if __name__ == "__main__":
    main()