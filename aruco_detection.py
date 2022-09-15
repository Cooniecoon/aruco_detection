#! /usr/bin/python3
import numpy as np
import cv2
import cv2.aruco as aruco
import sys
import math
from utils import *

def pose_esitmation(frame, aruco_dict_type, marker_size):


    aruco_dict = aruco.Dictionary_get(aruco_dict_type)
    parameters = aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = aruco.detectMarkers(frame, aruco_dict,parameters=parameters)

    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):

            ID=int(ids[i])

            aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            # aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.05)
            cv2.circle(frame,(int(corners[i][0][1][0]),int(corners[i][0][1][1])),4,0,-1)
        cv2.putText(frame,f"{ID}",(int(corners[0][0][1][0]-20),int(corners[0][0][1][1])-20),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),2)



    return frame


device_path = '/dev/video0'
cap = cv2.VideoCapture(device_path,cv2.CAP_V4L2)
aruco_dict_type = ARUCO_DICT["DICT_6X6_250"]
marker_size=0.04
dictionary = aruco.getPredefinedDictionary(aruco_dict_type)
while cap.isOpened():
    ret, frame = cap.read()
    output = pose_esitmation(frame, aruco_dict_type, marker_size)
    cv2.imshow('Estimated Pose', output)
    cv2.waitKey(1)

