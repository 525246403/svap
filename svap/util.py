#coding=utf-8
import cv2
import numpy
import os
from PIL import Image, ImageDraw, ImageFont

#convert mxnet box to rect 
def box2rect(box):
    x = box[0]
    y = box[1]
    w = box[2] - box[0]
    h = box[3] - box[1] 
    return (x,y,w,h)
    
#convert mxnet box to square 
def rect2square(x,y,w,h):
    c_x = x + int(w/2)
    c_y = y + int(h/2)
    ex_w = w
    if w < h:
       ex_w = h 
    ex_w = int(ex_w / 4) * 4
    rx = c_x - int(ex_w/2)  
    ry = c_y - int(ex_w/2)
    if rx < 0 :
        rx = 0
    if ry < 0 :
        ry = 0
    return (rx,ry,ex_w,ex_w)
    
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, numpy.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
    
    
    
def getValidRect(img_w,img_h, x,y,w,h):
    #get rectangle
    left = x if x > 0 else 0
    top = y if y > 0 else 0
    right = (x + w) if (x + w) < img_w else img_w
    bottom = (y + h) if (y + h) < img_h else img_h
    return (left, top ,right, bottom)
    
    
def makedir(filename):
    file_dir = os.path.split(filename)[0]
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
        
        
def extend_image(img,x,y,w,h,ratio = 0.5):
    img_h, img_w = img.shape[:2]  
    extend_x0 = int(x - ratio*w)
    extend_y0 = int(y - ratio*h)
    extend_x1 = int(x + (1+ratio)*w ) 
    extend_y1 = int(y + (1+ratio)*h )
    if extend_x0 < 0:
        extend_x0 = 0
    if extend_y0 < 0:
        extend_y0 = 0 
    if extend_x1 > img_w:
        extend_x1 = img_w
    if extend_y1 > img_h:
        extend_y1 = img_h      
    
    crop_img = img[extend_y0:extend_y1,extend_x0:extend_x1]     
    return crop_img  
