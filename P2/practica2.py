import numpy as np 
import cv2 as cv 
import os

def detectarPupilaIris(inImage):

    imagen_suavizada = cv.medianBlur(inImage, 5)

    imagen_color_pupila = cv.cvtColor(inImage, cv.COLOR_GRAY2BGR)
    imagen_color_iris = cv.cvtColor(inImage, cv.COLOR_GRAY2BGR)
    

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
            cv.circle(imagen_color_pupila, (i[0], i[1]), i[2], (0, 255, 0), 2)
            
     # Convertir las coordenadas a enteros
    if iris is not None:
        iris = np.uint16(np.around(iris))

        # Dibujar los círculos detectados
        for i in iris[0, :]:
            # Dibujar el contorno del círculo
            cv.circle(imagen_color_iris, (i[0], i[1]), i[2], (0, 255, 0), 2)

    return imagen_color_pupila,imagen_color_iris 

def detectarBrillosPupila(inImage):    
    cv.GaussianBlur(inImage,(5,5),0)
    contraste = np.array(255/np.log (1 + np.max(inImage))*np.log(1+inImage))
    contraste_pupila = np.logical_not(np.where(contraste > 160, 1, 0)).astype(np.uint8)
    
    contornos_brillos = []
    
    contornos, _ = cv.findContours(contraste_pupila, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    
    for c in contornos:
        if cv.contourArea(c) < 500:
            cv.drawContours(contraste_pupila,[c], 0, 255, 1)

    contornos_brillos.append(contraste_pupila)

    brillos = cv.bitwise_or(contornos_brillos[0], inImage)

    return brillos
    
def detectarPestañas(inImage):
    cv.GaussianBlur(inImage, (5, 5), 0)
    
    canny = cv.Canny(inImage, 50, 150)
    contornos, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contornos_pestanas = [c for c in contornos if cv.contourArea(c) < 150]
    
    pestanas = cv.cvtColor(inImage.copy(),cv.COLOR_GRAY2BGR)
    cv.drawContours(pestanas, contornos_pestanas, -1, (255, 255, 255), 2)
    
    return pestanas
    
    # sobel_x = cv.Sobel(inImage, cv.CV_64F, 1, 0, ksize=3)
    # sobel_y = cv.Sobel(inImage, cv.CV_64F, 0, 1, ksize=3)
    # magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    # magnitude = np.uint8(magnitude)
    # _, binary_image = cv.threshold(magnitude, 160, 255, cv.THRESH_BINARY)
    # contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # pestanas = inImage.copy()
    # cv.drawContours(pestanas,contours,-1,255,2)
    
    # return pestanas

def detectarEsclerotica(inImage): 
    # imagen_suavizada = cv.medianBlur(inImage, 5)
    # imagen_original = inImage.copy()
    
    # iris = cv.HoughCircles(imagen_suavizada,
    #                        cv.HOUGH_GRADIENT,
    #                        dp=1, 
    #                        minDist=50, 
    #                        param1=100, 
    #                        param2=35, 
    #                        minRadius=30, 
    #                        maxRadius=100)
    
    # if iris is not None:
    #     iris_contorno = np.uint16(np.around(iris))
    #     mask_iris = np.zeros_like(imagen_suavizada)
        
    #     for i in iris_contorno[0, :]:
    #         cv.circle(mask_iris, (i[0], i[1]), i[2], (255), 2)
            
    #     mask_alrededor_iris = cv.dilate(mask_iris, None, iterations=2)

    #     mask_esclerotica = cv.bitwise_not(mask_alrededor_iris)

    #     region_esclerotica = cv.bitwise_and(imagen_suavizada, imagen_suavizada, mask=mask_esclerotica)

    #     _, esclerotica_umbralizada = cv.threshold(region_esclerotica, 170, 255, cv.THRESH_BINARY)

    #     contornos, _ = cv.findContours(esclerotica_umbralizada, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #     for contorno in contornos:
    #         cv.drawContours(imagen_original, [contorno], 0, (0, 255, 0), 2)
            
    # return imagen_original
    
    #----------------------------------------K-MEANS----------------------------------
    
    # image_blur = cv.GaussianBlur(inImage,(7,7),0)
    # img_eq = cv.equalizeHist(image_blur)
    # img_filtered = cv.bilateralFilter(img_eq, 9, 75, 75)
    # img_float32 = np.float32(img_filtered)
    # img_flat = img_float32.flatten().reshape((-1, 1))
    # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # k = 2

    # _, labels, centers = cv.kmeans(img_flat, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # labels = labels.reshape(image_blur.shape)
    
    # binary_img = np.uint8(labels == 1) * 255
    
    # contours, _ = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # largest_contour = max(contours, key=cv.contourArea)
    
    # img_result = cv.drawContours(inImage.copy(), [largest_contour], -1, (0, 255, 0), 2)
    
    # return img_result
    
    #------------------------------------Umbralización adaptativa-------------------------
    _, umbral = cv.threshold(inImage,240, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    kernel = np.ones((5, 5), np.uint8)
    umbral = cv.morphologyEx(umbral, cv.MORPH_OPEN, kernel, iterations=2)
    
    contornos, _ = cv.findContours(umbral, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contornos_filtrados = [contorno for contorno in contornos if cv.contourArea(contorno) > 500]
    
    outImage = inImage.copy()
    cv.drawContours(outImage, contornos_filtrados, -1, (255, 255, 255), 2)

    return outImage

   
def main():
    images = os.listdir("entradas/")

    for image in images:
        name = os.path.basename(image) 

        inImage = cv.imread("entradas/" + name,cv.IMREAD_GRAYSCALE)
        assert inImage is not None, "Error: No se pudo cargar la imágen"

    
        pupila,iris = detectarPupilaIris(inImage)
        brillos = detectarBrillosPupila(inImage)
        esclerotica = detectarEsclerotica(inImage)
        pestañas = detectarPestañas(inImage)

        cv.imwrite("pupila/" + name ,pupila)
        cv.imwrite("iris/" + name ,iris)
        cv.imwrite("esclerotica/" + "esclerotica_" + name ,esclerotica)
        cv.imwrite("artefactos/" + "brillos_" + name ,brillos)
        cv.imwrite("artefactos/" + "pestanas_" + name, pestañas)

if __name__ == "__main__":
    main()