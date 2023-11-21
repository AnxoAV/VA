import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt
import math
import scipy.signal

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

    hist = cv.calcHist([np.uint8(inImage * 255)],[0],None,[nBins],[0,256])
    hist_acum = np.cumsum(hist)

    t = ((hist_acum/(height*width))*255)

    outImage = t[np.uint8(inImage * 255)]

    return outImage

#Filtro espacial de suavizado
def filterImage(inImage, kernel):
    P, Q = kernel.shape
    height, width = inImage.shape[:2]

    inImage_padded = np.pad(inImage, ((P // 2, P // 2), (Q // 2, Q // 2)), mode='edge')

    outImage = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            outImage[i, j] = np.sum(inImage_padded[i:i+P, j:j+Q] * kernel)

    return outImage

#Calculo de kernel Gaussiano
def gaussKernel1D(sigma):
    N = int(2 * np.ceil(3 * sigma) + 1)
    kernel = np.zeros(N)
    centro = (N - 1) // 2

    for i in range(N):
        #kernel[i] = np.exp(-(i - centro) ** 2) / (2 * sigma **2)  PREGUNTAR
        kernel[i] = (1.0 / (math.sqrt(2 * math.pi * sigma))) * math.exp(-((i-centro) ** 2)) / (2 * sigma ** 2)

    kernel /= np.sum(kernel)

    return kernel

#Filtro de suavizado Gaussiano
def gaussianFilter(inImage,sigma):
    kernel = gaussKernel1D(sigma)
    kernel2D = kernel[np.newaxis,:]
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

def generarImagen():
    matriz = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0],
                       [0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                       [0,0,1,1,1,1,1,0,0,0,0,1,1,1,0,0],
                       [0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0],
                       [0,0,0,1,1,0,0,0,0,1,1,1,1,1,0,0],
                       [0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0],
                       [0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0],
                       [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
                       [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],
                       [0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0],
                       [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
                       [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
                       [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                    ])

    matriz2 = np.array([
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0],
        [0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0],
        [0,0,0,1,1,1,1,0,1,0,0,1,1,1,0,0],
        [0,0,1,1,1,1,0,1,1,1,1,1,1,0,0,0],
        [0,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0],
        [0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
        [0,1,0,0,0,0,1,0,1,1,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
        [0,1,1,0,0,0,0,1,1,1,0,1,0,0,0,0],
        [0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ])

    matriz3 = np.array([
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0],
        [0,0,1,1,1,1,0,0,1,1,1,1,1,1,0,0],
        [0,0,1,1,1,1,0,0,1,1,1,1,1,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
        [0,0,1,1,0,0,0,0,0,1,1,1,1,0,0,0],
        [0,0,1,1,0,0,0,0,0,1,1,1,1,0,0,0],
        [0,0,0,1,0,0,0,0,0,1,1,1,1,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ])

    matriz4 = np.array([
        [1,1,1,1,1,1,1],
        [1,1,1,0,0,1,1],
        [1,1,0,0,0,0,1],
        [1,1,0,0,0,0,1],
        [1,1,1,0,0,1,1],
        [1,1,1,0,1,1,1],
        [1,1,1,1,1,1,1]
    ])

    matriz = matriz * 255
    matriz2 = matriz2 * 255
    matriz3 = matriz3 * 255
    matriz4 = matriz4 * 255
    cv.imwrite('entradas/hit_or_miss2.png',matriz4,[cv.IMWRITE_PNG_COMPRESSION,0])

#Función auxiliar para obtener grupos de tuplas en la función erode
def agrupar_tuplas(lista, tamano_grupo):
    grupos = []
    grupo_actual = []
    
    for elemento in lista:
        grupo_actual.append(elemento)
        
        if len(grupo_actual) == tamano_grupo:
            grupos.append(grupo_actual)
            grupo_actual = []

    return grupos

#Función auxiliar que devuelve la lista de coordenadas en donde existe un 1
def get1(matrix):
    filas,columnas = matrix.shape[:2]
    list1 = []
    for i in range(filas):
        for j in range(columnas):
            if matrix[i][j] == 1:
                list1.append((i,j))
    
    return list1

#Operadores morfológicos: erosión
def erode(inImage,SE,center = []):
    height,width = inImage.shape[:2]
    P,Q = SE.shape

    listCoords = []
    tmpCoords = []
    newCoords = []

    inImageOnes = get1(inImage)
    SEOnes = get1(SE)

    outImage = np.zeros((height,width))

    if not center:
        center.append(P // 2)
        center.append(Q // 2)

        for i in range(len(SEOnes)):
            newCoords.append((SEOnes[i][0] - center[0],SEOnes[i][1] - center[1]))
        for i in range(len(inImageOnes)):
            tmpCoords.append(inImageOnes[i])
            for j in range(len(newCoords)):
                tmpCoords.append((newCoords[j][0] + inImageOnes[i][0], newCoords[j][1] + inImageOnes[i][1]))

        grupos = agrupar_tuplas(tmpCoords,len(newCoords) + 1)

        for i in range(len(grupos)):
            cnt = 0
            for j in range(1,len(newCoords)+1):
                if(grupos[i][j] in inImageOnes):
                    cnt = cnt + 1
            if cnt == len(newCoords):
                listCoords.append(grupos[i][0])

        for i in range(height):
            for j in range(width):
                if (i,j) in listCoords:
                    outImage[i][j] = 1
    else:
        centerx = center[0]
        centery = center[1]

        for i in range(len(SEOnes)):
            newCoords.append((SEOnes[i][0] - centerx,SEOnes[i][1] - centery))

        for i in range(len(inImageOnes)):
            tmpCoords.append(inImageOnes[i])
            for j in range(len(newCoords)):
                tmpCoords.append((newCoords[j][0] + inImageOnes[i][0], newCoords[j][1] + inImageOnes[i][1]))

        grupos = agrupar_tuplas(tmpCoords,len(newCoords) + 1)

        for i in range(len(grupos)):
            cnt = 0
            for j in range(1,len(newCoords)+1):
                if(grupos[i][j] in inImageOnes):
                    cnt = cnt + 1
            if cnt == len(newCoords):
                listCoords.append(grupos[i][0])

        for i in range(height):
            for j in range(width):
                if (i,j) in listCoords:
                    outImage[i][j] = 1

    return outImage

#Operadores morfológicos: dilatación
def dilate(inImage,SE,center = []):
    height,width = inImage.shape[:2]
    P,Q = SE.shape

    listCoords = []
    newCoords = []

    inImageOnes = get1(inImage)
    SEOnes = get1(SE)

    outImage = np.zeros((height,width))

    if not center:
        center.append(P//2)
        center.append(Q//2)

        for i in range(len(SEOnes)):
            newCoords.append((SEOnes[i][0] - center[0],SEOnes[i][1] - center[1]))

        for i in range(len(newCoords)):
            for j in range(len(inImageOnes)):
                listCoords.append((newCoords[i][0] + inImageOnes[j][0], newCoords[i][1] + inImageOnes[j][1]))

        for i in range(height):
                for j in range(width):
                    if (i,j) in listCoords:
                        outImage[i][j] = 1        
    else:
        centerx = center[0]
        centery = center[1]

        for i in range(len(SEOnes)):
            newCoords.append((SEOnes[i][0] - centerx,SEOnes[i][1] - centery))

        for i in range(len(newCoords)):
            for j in range(len(inImageOnes)):
                listCoords.append((newCoords[i][0] + inImageOnes[j][0], newCoords[i][1] + inImageOnes[j][1]))

        for i in range(height):
                for j in range(width):
                    if (i,j) in listCoords:
                        outImage[i][j] = 1  

    return outImage

#Operadores morfológicos: erosión
def erode2(inImage, SE, center = []):

    if not center:
        center = [(SE.shape[0] // 2), (SE.shape[1] // 2)]
    else:
        center = [center[0], center[1]]

    outImage = np.copy(inImage)

    for i in range(inImage.shape[0]):
        for j in range(inImage.shape[1]):
            if inImage[i, j] == 1:
                for x in range(SE.shape[0]):
                    for y in range(SE.shape[1]):
                        if SE[x, y] == 1:
                            nx, ny = i + x - center[0], j + y - center[1]
                            if nx < 0 or ny < 0 or nx >= inImage.shape[0] or ny >= inImage.shape[1]:
                                continue
                            if inImage[nx, ny] != 1:
                                outImage[i, j] = 0
                                break
    return outImage

#Operadores morfológicos: dilatación
def dilate2(inImage, SE, center = []):

    if not center:
        center = [(SE.shape[0] // 2), (SE.shape[1] // 2)]
    else:
        center = [center[0], center[1]]

    outImage = np.copy(inImage)

    for i in range(inImage.shape[0]):
        for j in range(inImage.shape[1]):
            if inImage[i, j] == 1:
                for x in range(SE.shape[0]):
                    for y in range(SE.shape[1]):
                        if SE[x, y] == 1:
                            nx, ny = i + x - center[0], j + y - center[1]
                            if nx < 0 or ny < 0 or nx >= inImage.shape[0] or ny >= inImage.shape[1]:
                                continue
                            outImage[nx, ny] = 1
    return outImage

#Operadores morfológicos: apertura
def opening(inImage, SE, center = []):
    outImageTmp = erode(inImage,SE,center)
    outImage = dilate(outImageTmp,SE,center)

    return outImage

#Operadores morfológicos: cierre   
def closing(inImage, SE, center = []):
    outImageTmp = dilate(inImage,SE,center)
    outImage = erode(outImageTmp,SE,center)

    return outImage

def getComp(inImage):
    height,width = inImage.shape[:2]

    outImage = 1 - inImage

    return outImage

def intersec(hit,miss):
    row, col = hit.shape

    outImage = np.ones((row,col),dtype=np.uint8)

    for i in range(row):
        for j in range(col):
            if hit[i,j] == miss[i,j]:
                outImage[i,j] = hit[i,j]

    return outImage

#Transformada Hit-or-Miss
def hit_or_miss(inImage, objSEj, bgSE, center = []):
    P,Q = objSEj.shape[:2]

    for i in range(P):
        for j in range(Q):
            if (objSEj[i][j] == 1 and bgSE[i][j] == 1):
                print("Error: elementos estructurates incoherentes")
                return

    inImageComp = getComp(inImage)

    hit = getComp(erode(inImageComp,objSEj,[0,1]))
    miss = erode(inImage,objSEj,[1,0])

    outImage = intersec(hit,miss)

    return outImage

#Operadores de primera derivada
def gradientImage(inImage, operator):
    if operator == 'Roberts':
        op_gx = np.array([
            [-1,0],
            [0,1]
        ])

        op_gy = np.array([
            [0,-1],
            [1,0]
        ])
    elif operator == 'CentralDiff':
        op_gx = np.array([
           [-1,0,1] 
        ])

        op_gy = np.array([
            [-1],
            [0],
            [1]
        ])
    elif operator == 'Prewitt':
        op_gx = np.array([
            [-1,0,1],
            [-1,0,1],
            [-1,0,1]
        ])

        op_gy = np.array([
            [-1,-1,-1],
            [0,0,0],
            [1,1,1]
        ]) 
    elif operator == 'Sobel':
        op_gx = np.array([
            [-1,0,1],
            [-2,0,2],
            [-1,0,1]
        ]) 

        op_gy = np.array([
            [-1,-2,-1],
            [0,0,0],
            [1,2,1]
        ]) 

    else:
        raise ValueError("Operador no válido. Debe ser Roberts, CentralDiff, Prewitt o Sobel")

    gx = filterImage(inImage,op_gx)
    gy = filterImage(inImage,op_gy)

    return gx,gy

def magnitud(gx,gy):
    height,width = gx.shape[:2]

    outImage = np.zeros((height,width))

    for i in range(height):
        for j in range(width):
            outImage[i,j] = math.sqrt(pow(gx[i,j],2) + pow(gy[i,j],2))

    return outImage

#Filtro Laplaciano de Gaussiano
def LoG(inImage, sigma):

    kernel = np.array([
        [0,-1,0],
        [-1,4,-1],
        [0,-1,0]
    ])
    
    tmpImage = gaussianFilter(inImage,sigma)

    outImage = filterImage(tmpImage,kernel)

    return outImage

#Función auxiliar para calcular la supresión no máxima en Canny
def notMaxSupp(mag,direc):
    mag_supp = np.zeros_like(mag)

    for i in range(1,mag.shape[0]-1):
        for j in range(1,mag.shape[1]-1):
            direction = direc[i,j]
            m1,m2 = mag[i+1,j],mag[i-1,j]

            if((direction <= np.pi/4 and direction >= -np.pi/4) or (direction >= 3*np.pi/4) or (direction <= -3*np.pi/4)):
                m1,m2 = mag[i,j+1],mag[i,j-1]

            if mag[i,j] >= m1 and mag[i,j] >= m2:
                mag_supp[i,j] = mag[i,j]
    
    return mag_supp

#Detector de bordes de Canny
def edgeCanny(inImage,sigma,tlow,thigh):
    #Paso 1: Aplicamos filtro gaussiano
    tmpImage = gaussianFilter(inImage,sigma)

    #Paso 2: Obtener componentes gx y gy de la imagen y calcular la magnitud y la dirección
    gx,gy = gradientImage(inImage,'Sobel')
    mag = np.sqrt(gx**2 + gy**2)
    direc = np.arctan2(gy,gx)

    #Paso 3: Supresión no máxima
    mag_supp = notMaxSupp(mag,direc)

    #Paso 4: Umbral de histéresis
    strong = mag_supp > thigh 
    weak = (mag_supp >= tlow) & (mag_supp <= thigh)

    outImage = np.zeros_like(mag_supp)

    outImage[strong] = 1

    for i in range(1, outImage.shape[0] - 1):
        for j in range(1 , outImage.shape[1] - 1):
            if weak[i,j]:
                if(outImage[i+1,j-1:j+2].max() or
                   outImage[i,j-1:j+2].max() or
                   outImage[i-1,j-1:j+2].max()):

                   outImage[i,j] = 1

    return outImage 

#---------TESTS---------
#Test para probar el algoritmo de alteración del rango dinámico
def testAdjustIntensity():
    inImage = cv.imread('entradas/grays.png',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imágen"

    inImageNorm = inImage / 255.0

    outImage = adjustIntensity(inImageNorm,[],[0,1])

    cv.imwrite('salidas/imagen_rdinamico.png',np.uint8(outImage * 255))

#Test para probar el algoritmo de ecualización de histograma
def testEqualizeIntensity():
    inImage = cv.imread('entradas/tucan.jpeg',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imágen"

    inImageNorm = inImage / 255.0

    outImage = equalizeIntensity(inImageNorm) #Se pueden cambiar los nBins poniendo un segundo parámetro, por defecto nBins = 256

    cv.imwrite('salidas/imagen_ecualizada.png',outImage)

#Test para probar el funcionamiento del filtrado espacial mediante convolución
def testFilterImage():
    inImage = cv.imread('entradas/chica.jpeg',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imágen"

    inImageNorm = inImage / 255.0

    kernel = np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ])/9.0

    outImage = filterImage(inImageNorm,kernel)

    cv.imwrite('salidas/imagen_filtrada.jpg',np.uint8(outImage * 255))

#Test para probar el suavizado Gaussiano bidimensional
def testgaussianFilter():
    inImage = cv.imread('entradas/chica.jpeg',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imágen"

    inImageNorm = inImage / 255.0
    sigma = 1

    outImage = gaussianFilter(inImageNorm,sigma)

    cv.imwrite('salidas/imagen_Gauss.png',np.uint8(outImage* 255))

#Test para probar el filtro de medianas
def testMedianFilter():
    inImage = cv.imread('entradas/ruidoimpulsional.jpeg',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imágen"

    inImageNorm = inImage / 255.0
    filterSize = 7

    outImage = medianFilter(inImageNorm,filterSize)

    cv.imwrite('salidas/imagen_medianas.png',np.uint8(outImage * 255))  

#Test para probar la erosión en una imagen
def testDilate():
    inImage = cv.imread('entradas/dilatacion.png',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imágen"

    kernel = np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ])

    center = [0,0]

    inImageNorm = inImage / 255
    outImage = dilate(inImageNorm,kernel)

    cv.imwrite('salidas/imagen_dilatacion.png',outImage * 255) 

#Test para probar la dilatación en una imagen
def testErode():
    inImage = cv.imread('entradas/dilatacion.png',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imágen"

    kernel = np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ])

    center = [0,0]

    inImageNorm = inImage / 255

    outImage = erode(inImageNorm,kernel)

    cv.imwrite('salidas/imagen_erosion.png',outImage * 255)  

#Test para probar la apertura en una imagen
def testOpening():
    inImage = cv.imread('entradas/dilatacion.png',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imágen"

    kernel = np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ])

    center = [0,0]

    inImageNorm = inImage / 255

    outImage = opening(inImageNorm,kernel)

    cv.imwrite('salidas/imagen_apertura.png',outImage * 255)

#Test para probar el cierre en una imagen
def testClosing():
    inImage = cv.imread('entradas/cierre.png',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imágen"

    kernel = np.array([
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ])

    center = [0,0]

    inImageNorm = inImage / 255

    outImage = closing(inImageNorm,kernel)

    cv.imwrite('salidas/imagen_cierre.png',outImage * 255) 

#Test para probar la transformada hit-or-miss
def testHitOrMiss():
    inImage = cv.imread('entradas/image.png',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imágen"
    
    inImageNorm = inImage // 255

    objSEj = np.array([
        [1,1],
        [0,1]
    ],dtype=np.uint8)

    bgSE = np.array([
        [0,0],
        [1,0]
    ],dtype = np.uint8)

    outImage = hit_or_miss(inImageNorm,objSEj,bgSE)

    cv.imwrite('salidas/imagen_hit_or_miss.png',outImage * 255) 

#Función auxiliar para normalizar una imagen al rango [-1,1]
def normalizeImage(inImage):
    min = np.min(inImage)
    max = np.max(inImage)

    inImageNorm = 2 * ((inImage - min) / (max - min)) -1 

    return inImageNorm

def normalizeSobel(inImage):
    min_val = inImage.min()  # Encuentra el valor mínimo en la imagen
    max_val = inImage.max()  # Encuentra el valor máximo en la imagen
    new_max = 2
    new_min = -2
    # Normaliza la imagen al nuevo rango [-2, 2]
    normalized_image = ((inImage - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min

    return normalized_image

#Test para probar el cálculo de gradiente de una imagen
def testGradientImage():
    inImage = cv.imread('entradas/circles.png',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imágen"

    inImageNorm = inImage / 255

    operator = 'CentralDiff'

    gx,gy = gradientImage(inImageNorm,operator)

    outImage = magnitud(gx,gy)

    cv.imwrite('salidas/imagen_gx_' + operator + '.png',gx * 255) 
    cv.imwrite('salidas/imagen_gy_' + operator + '.png',gy * 255)
    cv.imwrite('salidas/imagen_magnitud_' + operator + '.png',outImage * 255)

#Test para probar el filtro Laplaciano de Gaussiano
def testLoG():
    inImage = cv.imread('entradas/chica2.jpeg',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imágen"

    inImageNorm = normalizeImage(inImage)

    outImage = LoG(inImageNorm,0.8)

    cv.imwrite('salidas/imagen_LoG.png',outImage * 255) 

#Test para probar el detector de bordes de canny
def testCanny():
    inImage = cv.imread('entradas/matricula.png',cv.IMREAD_GRAYSCALE)
    assert inImage is not None, "Error: No se pudo cargar la imágen"

    inImageNorm = inImage / 255

    outImage = edgeCanny(inImageNorm,1.4,0.3,0.7)

    cv.imwrite('salidas/imagen_Canny.png',outImage * 255) 

def main():

    # testAdjustIntensity()

    # testEqualizeIntensity()

    # testFilterImage()

    # testgaussianFilter()
    
    # testMedianFilter()

    # testErode()

    # testDilate()

    # testOpening()

    # testClosing()
    
    # testHitOrMiss()

    # testGradientImage()

    # testLoG()

    testCanny()
    


if __name__ == "__main__":
    main()