import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

ex = int(0)
ey = int(0)
ew = int(0)
eh = int(0)
ex2 = int(0)
ey2 = int(0)
eh2 = int(0)
ew2 = int(0)

inicio = 0
fim = 0
cont = 0
t = 0

def validar_camera(mirror, img):
    if mirror:
        img = cv2.flip(img, 1)
    return img


face_cascata = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
olho_cascata = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FPS, 30)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 120);
while (True):
    # captura frame por frame
    ok, frame = cap.read(5)
    if ok == False:
        break
    else:
        img = cv2.flip(frame, 1)
        img_original = img.copy()
        cv2.imshow('teste', img_original)
        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascata.detectMultiScale(img_cinza, 1.3, 5)
        centroX = (float)(0)
        #print(faces)
        for (x, y, w, h) in faces:
            img_face = img[y:y + h, x:x + w]
            if img_face is not None and img_face != []:
                img_face_original = img_face.copy()
                olhos = olho_cascata.detectMultiScale(img_face)
                centroX = (w - x)
                #print(olhos)
                #print(centroX)
                centroOlho = (float)(0)
                # cv2.waitKey(0)
                if olhos is not None and olhos != [] and len(olhos) >= 2:
                    if olhos[0] is not None and olhos[0] != []:
                        #print('olho[1]: ' + str(olhos[0]))
                        # cv2.waitKey(0)
                        ex, ey, ew, eh = olhos[0]
                    if olhos[1] is not None and olhos[1] != []:
                        #print('olho[1]: ' + str(olhos[1]))
                        # cv2.waitKey(0)
                        ex2, ey2, ew2, eh2 = olhos[1]
                if ex < ex2:
                    img_olho = img_face[ey:ey + eh, ex:ex + ew]
                    img_olhod = img_face[ey2:ey2 + eh2, ex2:ex2 + ew2]
                    img_cinza_olho = cv2.cvtColor(img_olhod.copy(), cv2.COLOR_BGR2GRAY)
                else:
                    img_olhod = img_face[ey:ey + eh, ex:ex + ew]
                    img_olho = img_face[ey2:ey2 + eh2, ex2:ex2 + ew2]
                    img_cinza_olho = cv2.cvtColor(img_olhod.copy(),cv2.COLOR_BGR2GRAY)

                if img_cinza_olho is not None:
                    blur = cv2.GaussianBlur(img_cinza_olho, (3, 3), 0)
                    # ret3,th3 = cv2.threshold(blur,20,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    ret3, th3 = cv2.threshold(blur, 55, 255, cv2.THRESH_BINARY)
                    cv2.imshow('threshold',th3)
                    mascara = np.ones((5, 5), np.uint8)
                    erosao = cv2.erode(th3, mascara, iterations=1)
                    masc = np.ones((5, 5), np.uint8) # 5,5
                    dilatacao = cv2.dilate(erosao, masc, iterations=1)
                    cann = cv2.Canny(dilatacao, 5, 30)
                    cv2.imshow('dilatacao', dilatacao)
                    cv2.imshow('canny', cann)
                    circles = cv2.HoughCircles(cann, cv2.HOUGH_GRADIENT, 1, 5, np.array([]), 100, 10, 1, 20) #4 por 5
                    if circles is not None and circles != []:
                        a, b, c = circles.shape
                        cv2.putText(img_face_original, "Aberto", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),2)
                        if t==1:
                            inicio = 0
                            fim = 0
                            t = 0
                        for i in range(b):
                            cv2.circle(cann, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 3,
                                       cv2.LINE_AA)
                    else:
                        if t==0:
                            inicio = time.time()
                            t = 1
                            cont = cont + 1
                        fim = inicio + fim

                        cv2.putText(img_face_original, "Fechado" , (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 0, 0), 2)

                    cv2.putText(img_face_original, "" + (str)(cont), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
                    cv2.putText(img_face_original, "" + (str)(fim), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)

                if img_face is not None and img_face != []:
                    cv2.imshow('face', img_face_original)

                if img_olho is not None and img_olho!=[]:
                    cv2.imshow('olho', img_olho)
                if img_olhod is not None and img_olhod !=[]:
                    cv2.imshow('olhod', img_olhod)


        key = cv2.waitKey(1)

        if key == ord('p'):
            cv2.imwrite('olho.jpg', img_olho)
        if key == ord('f'):
            cv2.imwrite('face.jpg', img_face)
        if key == ord('q'):
            break;

cap.release()
cv2.destroyAllWindows()
