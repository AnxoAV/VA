import numpy as np 
import cv2 as cv 
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

    pupila = cv.HoughCircles(
        imagen_suavizada,
        cv.HOUGH_GRADIENT,
        dp=1,  # Resolución acumulativa del detector
        minDist=50,  # Distancia mínima entre los centros de los círculos
        param1=100,  # Umbral del detector de bordes (Canny)
        param2=30,  # Umbral del acumulador de Hough
        minRadius=15,  # Radio mínimo del círculo
        maxRadius=40  # Radio máximo del círculo
    )

    iris = cv.HoughCircles(
        imagen_suavizada,
        cv.HOUGH_GRADIENT,
        dp=1,  # Resolución acumulativa del detector
        minDist=50,  # Distancia mínima entre los centros de los círculos
        param1=100,  # Umbral del detector de bordes (Canny)
        param2=35,  # Umbral del acumulador de Hough
        minRadius=30,  # Radio mínimo del círculo
        maxRadius=100 # Radio máximo del círculo
    )

    # Convertir las coordenadas a enteros
    if pupila is not None:
        pupila = np.uint16(np.around(pupila))

        # Dibujar los círculos detectados
        for i in pupila[0, :]:
            # Dibujar el contorno del círculo
            cv.circle(imagen_color, (i[0], i[1]), i[2], (0, 255, 0), 2)

     # Convertir las coordenadas a enteros
    if iris is not None:
        iris = np.uint16(np.around(iris))

        # Dibujar los círculos detectados
        for i in iris[0, :]:
            # Dibujar el contorno del círculo
            cv.circle(imagen_color, (i[0], i[1]), i[2], (0, 0, 255), 2)

    return imagen_color    

# Lista de cosas probadas:
#   1 - Umbralización con detección de contornos.
#   2 - Umbralización adaptativa con detección de contornos.
#   3 - Transformada de Hough y formar una elipse

def detectarEsclerotica(inImage): 
    
    canny = cv.Canny(inImage,80,200)
    _, thresh = cv.threshold(canny, 200, 255, cv.THRESH_BINARY_INV)
    
    kernel = np.ones((5,5))
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 2)
    
    contornos, _ = cv.findContours(opening,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    
    outImage = cv.cvtColor(inImage.copy(), cv.COLOR_GRAY2BGR)
    cv.drawContours(outImage,contornos, -1, (255, 0, 0))
    
    return outImage
    

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

    
        pupila = detectarPupila(inImage)
        esclerotica = detectarEsclerotica(inImage)

        cv.imwrite("salidas/" + name ,pupila)
        cv.imwrite("salidas/" + "esclerotica_" + name ,esclerotica)

        
    
if __name__ == "__main__":
    main()