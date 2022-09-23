import os
import numpy as np
import cv2 as cv

people=[]
for i in os.listdir(r'C:\Users\saad\Documents\pypractice\photos\training'):
    people.append(i)

dir = r'C:\Users\saad\Documents\pypractice\photos\training'

haar_cascade = cv.CascadeClassifier('Faces/haar_face.xml') # stores the 33000 lines of xml code in a var

features = []   # stores features i.e number of faces
labels = []     # stores identity of those faces

def create_train():
    for person in people:                            # for each folder in folders of people
        path = (os.path.join(dir, person))           # join that folder path with parent folder
        label = people.index(person)                 # returns index of the persons and stores it in label

        for img in os.listdir(path):                  # for every image in each persons folder
            img_path = os.path.join(path,img)        # join that img's path with the parent path
            img_array = cv.imread(img_path)          # read every image from that folder and create an array of it
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY) # cvt image to gray scale
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors= 4)  # detect faces from the gray scale
            
            for (x,y,w,h) in faces_rect:             # for all the cordinates in the rectangle
                faces_roi = gray[y:y+h, x:x+w]       # cropping out the face
                features.append(faces_roi)           # add all faces to the features list
                labels.append(label)                 # add all the index holding names to the labels list, reducing the labels to numeric value to reduce strain on computer
                
create_train()

print('Training Donee........!')

features = np.array(features, dtype= 'object')                        # converting features list to nd arrays
labels = np.array(labels)                          # converting labels list to nd arrays

print(f'Length of feature is {len(features)}')
print(f'Length of labels is {len(labels)}')         

face_recognizer = cv.face.LBPHFaceRecognizer_create()  # initate facial recogniztion

# Train the Recognizer on features and labels list

face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')

np.save('features', features)
np.save('labels', labels)




