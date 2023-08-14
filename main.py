import cv2

img = cv2.imread('images/f5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cv2.CascadeClassifier('face.xml') #подгружаю файл с натренированной нейросетью

results = faces.detectMultiScale(gray, scaleFactor=1.104, minNeighbors=5) #возвращает координаты искомых обьектов
for (x, y, w, h) in results:
    cv2.circle(img, (x + (w // 2), y + (h // 2)), h // 2, (0, 255, 0), thickness=2)


cv2.imshow('Result', img)
cv2.waitKey(0)