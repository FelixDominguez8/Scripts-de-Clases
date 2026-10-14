import cv2
import numpy as np

def procesar_imagen(ruta_imagen):
    # 1. Cargar la imagen (Concepto: Imagen Digital RGB) [cite: 11]
    # OpenCV carga las imágenes en formato BGR por defecto
    img = cv2.imread(ruta_imagen)
    if img is None:
        print("Error: No se pudo cargar la imagen.")
        return

    # 2. Conversión a Escala de Grises (Reducción de Dimensión) 
    # Reduce la imagen de 3 canales (RGB) a solo 1 canal de brillo [cite: 20]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Normalización de Píxeles 
    # Escala los valores de 0-255 a un rango de 0-1 [cite: 21, 23]
    normalized = gray.astype(np.float32) / 255.0

    # 4. Ecualización de Histograma [cite: 30, 33]
    # Mejora el contraste redistribuyendo las intensidades [cite: 31, 32]
    equalized = cv2.equalizeHist(gray)

    # 5. Umbralización (OTSU Thresholding) [cite: 34, 35]
    # Binariza la imagen separando fondo y objeto automáticamente 
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. Suavizado (Gaussian Blur) [cite: 52, 54]
    # Reduce el ruido de alta frecuencia usando un kernel [cite: 52, 53]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 7. Detección de Bordes (Canny) [cite: 51, 62]
    # Identifica transiciones abruptas de intensidad [cite: 59, 62]
    edges = cv2.Canny(gray, 100, 200)

    # 8. Operaciones Morfológicas (Dilatación y Erosión) [cite: 63, 64]
    # Se aplican sobre imágenes binarias para mejorar características [cite: 64]
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(otsu, kernel, iterations=1) # Elimina ruido pequeño [cite: 64]
    dilation = cv2.dilate(otsu, kernel, iterations=1) # Rellena huecos [cite: 65]

    # Mostrar resultados
    cv2.imshow('Original', img)
    cv2.imshow('Escala de Grises', gray)
    cv2.imshow('Ecualizada (Contraste)', equalized)
    cv2.imshow('Binarizacion OTSU', otsu)
    cv2.imshow('Bordes Canny', edges)
    
    print("Procesamiento completado. Presiona cualquier tecla para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

procesar_imagen('cheems.jpeg')