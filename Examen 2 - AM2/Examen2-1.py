import cv2

def procesar():
    imagen = cv2.imread("hello2.png")

    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    _, imagen = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    imagen = cv2.GaussianBlur(imagen, (5, 5), 0)

    imagen = cv2.Canny(imagen, 100, 200)

    imagen = cv2.bitwise_not(imagen)

    cv2.imshow('Final', imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

procesar()