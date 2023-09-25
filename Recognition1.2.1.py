# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 17:09:11 2021

@author: Avery Booth
"""
import math

import cv2
from pil import Image
from lobe import ImageModel



HOST = "127.0.0.1"
PORT = 65432


#----------------------------------------------------------------------------------------------
# set up for the Facial Recognition ML Model

modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#----------------------------------------------------------------------------------------------
# Loading
model = ImageModel.load(r'./model/Subset TFLite') # Loads the Mask Recognition Model
cap = cv2.VideoCapture(0) #Captures Live Video From Computer Camera

#----------------------------------------------------------------------------------------------
# Variable Set Up
hasFrame, frame = cap.read()

frame_count = 0
tt_opencvDnn = 0

center_point_previous=[] # Used to store the center points of faces in the frame before the one being Analyzed

Instances={} # used to store the number and location of Faces detected in the current Frame

InstanceID=0# Id assigned to each Instance

count=0

NOMASKED=[] #List containing the ID's of instances who have been identified as not having a mask on

conf_threshold=0.50 # the minimum percent of confidence the facial recognition 
                    # needs to have in what it see's being a Face before adding it to Instances
                    

#----------------------------------------------------------------------------------------------
#Capturing and PreProcessing of Frame

while True:
    hasFrame, frame = cap.read() # takes a single frame from the video Feed
    InstanceID=InstanceID+1 
    if not hasFrame: # checks to make sure a frame has been Captured
        break
    frame_count += 1 # count the amount of Frames gone Through
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    
#----------------------------------------------------------------------------------------------
    blob = cv2.dnn.blobFromImage(
        frameOpencvDnn, 1.0, (299, 299), [104, 117, 123], True, False,
    )
    net.setInput(blob)
    detections = net.forward()
#----------------------------------------------------------------------------------------------
# Goes through Every Face detected and appends any info regarding the four frame points around the face and the faces Centerpoint to centerpoints

    center_point =[] # Used to store locational data and centerpoints of faces from Current Frame 
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            cx=int((x1+x2)/2) #Calculating Center x cordinate
            cy=int((y1+y2)/2) #Calculating Center y cordinate

            center_point.append((cx,cy,x1,y1,x2,y2))



#----------------------------------------------------------------------------------------------
# Used to Track Faces By comparing the Center points of faces from the current frame to Ones From the Previos ones


    if frame_count <= 2: # used to give the program A Chance to record a set of Previous center points before  beggining to compare
       
       for pt in center_point:
           for pt2 in center_point_previous:
               distance=math.hypot(pt2[0]-pt[0], pt2[1]-pt[1])
               if distance < 20:
                  
                   Instances[InstanceID]= pt # adds location and Centerpoint Data to Instances
                   InstanceID =InstanceID+ 1 # increments Instance ID so another with the same ID dosen't Pop up
    if frame_count > 2:
        Instances2 =Instances.copy()
        center_point2=center_point.copy()
        
        for InstanceID, pt2 in Instances2.items():
            exists =False
            for pt in center_point2:
                    distance=math.hypot(pt2[0]-pt[0], pt2[1]-pt[1])
                    if distance < 20:
                        Instances[InstanceID] = pt # adds new location and Centerpoint Data to existing Instance 
                        exists=True
                        center_point.remove(pt) #removes no longer neede data from Centerpoint
                        continue

            if exists == False:
                    Instances.pop(InstanceID) #removes instance if no Longer Detected
                    if InstanceID in NOMASKED:
                        NOMASKED.remove(InstanceID)# Removes instance from Nomask log in not detected
        for pt in center_point:
           Instances[InstanceID]=pt # adds location and Centerpoint Data to Instances of new instance
           InstanceID =InstanceID + 1 # increments Instance ID so another with the same ID dosen't Pop up


#----------------------------------------------------------------------------------------------
# Goes through Every Instance in Instances ID and clips out the face with the coordinates 
# provided by the instance and then passes it into the Mask Recognition Model before taking the 
# results to detemine if the border to be drawn around the face should be green (For Having A Mask)
# or Red (Having no mask)
    for InstanceID, pt in Instances.items():
        
        height=pt[5]-pt[3]#
        if height > 10:#
            h1=height * 0.15 #
            h1=int(h1) #
            h2=height * 0.1 #
            h2=int(h2) #
            color1 = frame[pt[3]+h2:pt[5]-h1, pt[2]:pt[4]]# new slicing op
        else:#
            color1 = frame[pt[3]:pt[5], pt[2]:pt[4]]# new slicing op
        cv2.imshow('stuff1', color1)
        
        if InstanceID in NOMASKED:
           face_frame = cv2.resize(color1, (600, 600))

           face_frame = Image.fromarray(face_frame)
           q=(model.predict(face_frame))
           q=q.prediction
           label = "Mask" if q =="Mask" else "Not Masked"
           color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
           cv2.rectangle(frameOpencvDnn,(pt[2], pt[3]),(pt[4], pt[5]),color,int(round(frameHeight / 200)),2)
           cv2.putText(frameOpencvDnn, str(InstanceID), (pt[0],pt[1]-20), 0, 1, (0,0,255),2)
         

        else:
            face_frame = cv2.resize(color1, (600, 600))

            face_frame = Image.fromarray(face_frame)
            q=(model.predict(face_frame))
            q=q.prediction

            if q != "Mask":
                NOMASKED.append(InstanceID)
            else:
                label = "Mask" if q =="Mask" else "Not Masked"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                cv2.rectangle(frameOpencvDnn,(pt[2], pt[3]),(pt[4], pt[5]),color,int(round(frameHeight / 200)),2)
                cv2.putText(frameOpencvDnn, str(InstanceID), (pt[0],pt[1]-20), 0, 1, (0,0,255),2)
                
#---------------------------------------------------------------------------------------------- 
# Displays Diagnostic Info         
    print("Current_Frame")
    print(frame_count)
    print("current Variable")   
    print(InstanceID)
    print("In no masked")
    print(NOMASKED)
    frameOpencvDnn= cv2.resize(frameOpencvDnn, (600, 600))
    cv2.imshow('stuff stuff', frameOpencvDnn)
    cv2.waitKey(1)
    center_point_previous=center_point.copy()
