import numpy as np
import cv2 as cv
import os

people=[]
for i in os.listdir(r'C:\Users\saad\Documents\pypractice\photos\training'):
    people.append(i)

haar_cascade = cv.CascadeClassifier('Faces/haar_face.xml') # stores the 33000 lines of xml code in a var
#features = np.load('features.npy')                         # loading the saved features,labels
#labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()  # initate facial recogniztion
face_recognizer.read('face_trained.yml')

print('\nWelcome to "Guess the Footballer" Challenge\n')

img_user = input('Enter the path to the Image of the footballer:\n')
img = cv.imread((img_user))

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Unidentified Person', gray)

# detect face

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)

    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (10,150), cv.FONT_HERSHEY_COMPLEX, 0.7, color= (0,255,0),thickness=2)
    cv.rectangle(img, (x,y),(x+w,y+h), (0,255,0), 2)

cv.imshow('Detected Face', img)    

cv.waitKey(0)



