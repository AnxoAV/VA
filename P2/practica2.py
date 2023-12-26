import numpy as np 
import cv2 as cv 
import os

def detectarPupila(inImage):

    imagen_suavizada = cv.medianBlur(inImage, 5)

    imagen_color = cv.cvtColor(inImage, cv.COLOR_GRAY2BGR)

    pupila = cv.HoughCircles(
        imagen_suavizada,
        cv.HOUGH_GRADIENT,
        dp=1,  
        minDist=50, 
        param1=100,  
        param2=30,  
        minRadius=15,  
        maxRadius=40  
    )

    iris = cv.HoughCircles(
        imagen_suavizada,
        cv.HOUGH_GRADIENT,
        dp=1, 
        minDist=50, 
        param1=100, 
        param2=35, 
        minRadius=30, 
        maxRadius=100 
    )

    # Convertir las coordenadas a enteros
    if pupila is not None:
        pupila = np.uint16(np.around(pupila))

        # Dibujar los círculos detectados
        for i in pupila[0, :]:
            # Dibujar el contorno del círculo
            cv.circle(imagen_color, (i[0], i[1]), i[2], (0, 255, 0), 2)
            brillos = detectarBrillosPupila(inImage,i[0], i[1], i[2])
            
     # Convertir las coordenadas a enteros
    if iris is not None:
        iris = np.uint16(np.around(iris))

        # Dibujar los círculos detectados
        for i in iris[0, :]:
            # Dibujar el contorno del círculo
            cv.circle(imagen_color, (i[0], i[1]), i[2], (0, 0, 255), 2)

    return imagen_color,brillos   

# Lista de cosas probadas:
#   1 - Umbralización con detección de contornos.
#   2 - Umbralización adaptativa con detección de contornos.
#   3 - Transformada de Hough y formar una elipse

def detectarEsclerotica(inImage): 
    
    # canny = cv.Canny(inImage,20,200)
    # _, thresh = cv.threshold(canny, 200, 255, cv.THRESH_BINARY_INV)
    
    # kernel = np.ones((5,5))
    # opening = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    
    # contornos, _ = cv.findContours(opening,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    
    # outImage = cv.cvtColor(inImage.copy(), cv.COLOR_GRAY2BGR)
    # cv.drawContours(outImage,contornos, -1, (255, 0, 0))
    
    #--------------------------------------------------------------------------
    
    # _, thresh = cv.threshold(inImage, 200, 255, cv.THRESH_BINARY_INV)
    
    # contornos, _ = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    
    # mascara_pestanas = np.zeros_like(inImage)
    
    # cv.drawContours(mascara_pestanas, contornos, -1, (255), thickness=cv.FILLED)
    
    # mascara_no_pestanas = cv.bitwise_not(mascara_pestanas)
    
    # resultado = cv.bitwise_and(inImage, inImage, mask=mascara_no_pestanas)
    
    # canny = cv.Canny(resultado,90,200)
    
    # -----------------------------------------------------
    
    # image_blur = cv.GaussianBlur(inImage,(5,5),0)
    
    # sobelx = cv.Sobel(image_blur,cv.CV_64F,1,0,ksize = 5)
    # sobely = cv.Sobel(image_blur,cv.CV_64F,0,1,ksize = 5)
    # edges = np.sqrt(sobelx**2 + sobely**2)
    

    
    # _, thresholded = cv.threshold(edges, 50, 255, cv.THRESH_BINARY)
    # contours, _ = cv.findContours(thresholded.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # largest_contour = max(contours, key=cv.contourArea)
    
    # img_result = cv.drawContours(inImage.copy(), [largest_contour], -1, (0, 255, 0), 2)
    
    # -------------------------------
    
    image_blur = cv.GaussianBlur(inImage,(7,7),0)
    img_eq = cv.equalizeHist(image_blur)
    img_filtered = cv.bilateralFilter(img_eq, 9, 75, 75)
    img_float32 = np.float32(img_filtered)
    img_flat = img_float32.flatten().reshape((-1, 1))
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 2

    _, labels, centers = cv.kmeans(img_flat, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape(image_blur.shape)
    
    binary_img = np.uint8(labels == 1) * 255
    
    contours, _ = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    largest_contour = max(contours, key=cv.contourArea)
    
    img_result = cv.drawContours(inImage.copy(), [largest_contour], -1, (0, 255, 0), 2)

    
    return img_result

def detectarBrillosPupila(inImage,centro_x,centro_y,radio):
    
    mascara_pupila = np.zeros_like(inImage)
    cv.circle(mascara_pupila, (centro_x, centro_y), radio, (255), thickness=cv.FILLED)
    region_pupila = cv.bitwise_and(inImage, mascara_pupila)
    _, thresh = cv.threshold(region_pupila, 200, 255, cv.THRESH_BINARY)
    contornos, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mascara_brillos = cv.cvtColor(inImage.copy(), cv.COLOR_GRAY2BGR)
    cv.drawContours(mascara_brillos, contornos, -1, (0,0,255), thickness=cv.FILLED)
    mascara_brillos_color = cv.resize(mascara_brillos, (inImage.shape[1], inImage.shape[0]))
    brillos = cv.addWeighted(mascara_brillos, 1, mascara_brillos_color, 0.5, 0)
    
    return brillos

    
def detectarPestañas(inImage):
    canny = cv.Canny(inImage,70,200)
    
    _,thresh = cv.threshold(canny,200,255,cv.THRESH_BINARY)
    
    contours, _ = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    
    outImage = cv.cvtColor(inImage.copy(),cv.COLOR_GRAY2BGR)
    
    cv.drawContours(outImage,contours, -1, (255,0,0),2)
    
    return outImage

def main():
    images = os.listdir("entradas/")

    for image in images:
        name = os.path.basename(image) 

        inImage = cv.imread("entradas/" + name,cv.IMREAD_GRAYSCALE)
        assert inImage is not None, "Error: No se pudo cargar la imágen"

    
        pupila,resultado = detectarPupila(inImage)
        esclerotica = detectarEsclerotica(inImage)
        pestañas = detectarPestañas(inImage)

        # cv.imwrite("salidas/" + name ,pupila)
        cv.imwrite("salidas/" + "esclerotica_" + name ,esclerotica)
        # cv.imwrite("salidas/" + "brillos_" + name ,resultado)
        # cv.imwrite("salidas/" + "pestaas_" + name, pestañas)

        
    
if __name__ == "__main__":
    main()