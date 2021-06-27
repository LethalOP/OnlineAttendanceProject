import face_recognition
import cv2
import os
import numpy as np
import datetime
from datetime import date
import pyautogui
import pyscreenshot as ImageGrab
import pandas as pd
import csv
#for screen recording
def screenRec():
    screenShot = pyautogui.screenshot()
    screenShot.save("Image.png")
    imageresol = screenShot.size

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frameRate = 30.0
    videoWriter = cv2.VideoWriter("video.mp4",fourcc,frameRate,imageresol)
    for i in range(300):
        screenShot = pyautogui.screenshot()
        numpyImageArray = np.array(screenShot)
        numpyImageArray = cv2.cvtColor(numpyImageArray,cv2.COLOR_BGR2RGB)
        videoWriter.write(numpyImageArray)
    cv2.destroyAllWindows()
    videoWriter.release()


classNames = []
images = []
path = 'known'
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0].upper())
print(classNames)
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
encodeListKnown = findEncodings(images)

#csv part
nameList = []
def MarkAttendance(name):
    with open('MarkAttendance.csv','r+') as f:
        myDataList = f.readlines()
        
        
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f' \n{name},{dtString}')
'''
def Students(studentsList):
   with open ('Students.csv', "w+", newline = "") as f:
    #csv_dict = [row for row in csv.DictReader(f)]
    #if len(csv_dict) == 0:
        print('csv file is empty')
        li = []
        df = pd.DataFrame(list())
       
        print(df)
        #df = pd.read_csv("Students.csv")
        df['Name']=pd.Series(studentsList)
        x = date.today()
        df[x] = ""
        for i in range(len(df.index)):
            for j in range(len(actual)):
                if df['Name'][i] == actual[j]:
                    print(i)
                    df.at[i,x] = 1

        
        print(li)
        df[x] = pd.Series(li)
        df.to_csv("Students.csv", index=False)
        print(df)

    else:
        print('csv is not empty')
        x = date.today()
        #for i in 
        df[x] = ""
        df.to_csv("Students.csv", index=False)
        print(df)

def writer(header, data, filename):
  with open (filename, "w", newline = "") as csvfile:
    movies = csv.writer(csvfile)
    movies.writerow(header)
    for x in range(len(data)):
      movies.writerow(data[x]+'\n')

def Students(lst):
    df = pd.read_csv("Students.csv")
    filename = 'Students.csv'
    df = pd.DataFrame(lst)
    x = date.today()
    #df['Date'] = pd.Series([x])
    print(df)
    data = df.iloc[:,0];
    header = [] 
    header = ['Name']
    header.append(x)
    writer(header,data,filename)

def Students(studentsList):
    df = pd.read_csv("Students.csv") 
    df.empty
'''
   
#Video part
cap = cv2.VideoCapture('video1.mp4')
for i in range(100): 
    sucess,img = cap.read()
    imgS = cv2.resize(img,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_AREA)
    print(imgS)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    
    for encodeFace,faceloc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        print(matches)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name  = classNames[matchIndex].upper()
            #print(name)
            print(faceloc)
            (top,right,bottom,left) = faceloc
            start_pt = (top,left)
            end_pt = (right,bottom)
            color = (255,0,0) #red
            thickness = 2
            cv2.rectangle(img,start_pt,end_pt,color,thickness)
            cv2.putText(img,name,(left+2,bottom+2),cv2.FONT_HERSHEY_PLAIN,1.2,(255,0,0),1)
            MarkAttendance(name)

            
        cv2.imshow("Video",img)#edited
        cv2.waitKey(1)#edited

actual = list(set(nameList))
print(actual) 
#Students(classNames)  
    
print('Attendance Complete')