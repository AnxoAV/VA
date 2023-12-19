import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt
import math
import scipy.signal
import os

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

def detectarPupila(inImage):

    imagen_suavizada = cv.medianBlur(inImage, 5)

    imagen_color = cv.cvtColor(inImage, cv.COLOR_GRAY2BGR)

    circulos_detectados = cv.HoughCircles(
        imagen_suavizada,
        cv.HOUGH_GRADIENT,
        dp=1,  # Resolución acumulativa del detector
        minDist=50,  # Distancia mínima entre los centros de los círculos
        param1=100,  # Umbral del detector de bordes (Canny)
        param2=30,  # Umbral del acumulador de Hough
        minRadius=15,  # Radio mínimo del círculo
        maxRadius=40  # Radio máximo del círculo
    )

    # Convertir las coordenadas a enteros
    if circulos_detectados is not None:
        circulos_detectados = np.uint16(np.around(circulos_detectados))

        # Dibujar los círculos detectados
        for i in circulos_detectados[0, :]:
            # Dibujar el contorno del círculo
            cv.circle(imagen_color, (i[0], i[1]), i[2], (0, 255, 0), 2)

    return imagen_color


def main():
    # inImage = cv.imread("entradasp2/aevar1.bmp",cv.IMREAD_GRAYSCALE)
    # assert inImage is not None, "Error: No se pudo cargar la imágen"
    
    # img_smooth = cv.GaussianBlur(inImage,(3,3),0)

    # laplacian = cv.Laplacian(img_smooth,cv.CV_64F)

    # laplacian = np.uint8(np.absolute(laplacian))

    # pupila = cv.Canny(inImage,80,200)

    # iris_pupila = cv.Canny(inImage,80,100)

    # outImage = adjustIntensity(laplacian/255,[],[0,1])

    # sobel_x = cv.Sobel(inImage,cv.CV_64F,1,0,ksize = 3)
    # sobel_y = cv.Sobel(inImage,cv.CV_64F,0,1,ksize = 3)

    # magnitud_sobel = np.sqrt(sobel_x**2 + sobel_y**2)

    # detectarIris(inImage)

    # cv.imwrite("salidasp2/chongpkr2_LoG.bmp",outImage * 255)
    # cv.imwrite("salidasp2/chongpkr2_Pupila.bmp",pupila)
    # cv.imwrite("salidasp2/chongpkr2_IrisPupila.bmp",iris_pupila)
    # cv.imwrite("salidasp2/chongpkr2_Sobel.bmp",np.uint8(magnitud_sobel))
    # cv.imwrite("salidasp2/chongpkr2_Circulos.bmp",detectarIris(inImage))

    images = os.listdir("entradas/")

    for image in images:
        name = os.path.basename(image) 

        inImage = cv.imread("entradas/" + name,cv.IMREAD_GRAYSCALE)
        assert inImage is not None, "Error: No se pudo cargar la imágen"

        outImage = cv.Canny(inImage,50,150)

        cv.imwrite("salidas/" + name ,detectarPupila(inImage))

        
    
if __name__ == "__main__":
    main()