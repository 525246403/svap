#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2

KEY_ESC     = 27
KEY_SPACE   = 32
KEY_CR      = 13

COLOR_GREEN = (0, 255, 0)
COLOR_RED   = (0, 0, 255)
COLOR_BLUE  = (255, 0, 0)

def cvRectangle(img, pt1, pt2, color=COLOR_GREEN, thickness=1):
    cv2.rectangle(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, thickness)
    
def cvRectangleR(img, box, color=COLOR_GREEN, thickness=1):
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[0]+box[2]), int(box[1]+box[3])), color, thickness) 
    
def cvZero(img):
    img[::] = 0