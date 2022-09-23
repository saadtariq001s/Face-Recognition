import cv2 as cv

img = cv.imread('photos/messi.jpg')
cv.imshow('Messi', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Messi', gray)

haar_cascade = cv.CascadeClassifier('Faces/haar_face.xml') # stores the 33000 lines of xml code in a var

faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.2, minNeighbors=2)

print(f'Number of Faces found are {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h),(0,0,255),2)

cv.imshow('faces', img)    


cv.waitKey(0)

