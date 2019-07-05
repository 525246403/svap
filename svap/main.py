# -*- coding: utf-8 -*-
import os
import sys
import json
import cv2
import pickle

from util import getValidRect,cv2ImgAddText
from celeb_rec import initial,process_video,destroy
from facerec_util import decode_feat
sys.path.append('./facerec')
from speedmeter import PrintNotifier


def get_celeba_db():
    info_file = '/home/ysten/denghui/svap/data/celeb_info.txt'
    feat_file = '/home/ysten/denghui/svap/data/celeb_feats.pkl2'
    find_name = ['周杰伦','陈冠希','余文乐','杜汶泽','黄秋生','钟镇涛']
    #find_name = ['梁家辉','周润发','黄秋生','刘德华']
    idList = []
    info_dict = {}
    with open(info_file,'r') as fr:
        for line in fr:
            vec = line.strip().split('\t')
            info_dict[vec[1]]=vec[0]

    for name in find_name:
        if name in info_dict.keys():
            idList.append(info_dict[name])
 
    pkl_file = open(feat_file, 'rb')
    feat_dict = pickle.load(pkl_file)
    face_db ={}
    for (i,id) in enumerate(idList):
        feats = feat_dict[id].values()
        face_db[id]=[find_name[i],feats]  
    #print(face_db)
    #decode feats
    for (id,[name,fs]) in face_db.items():
        feats = []
        for f in fs:
            feats.append(decode_feat(f))
        face_db[id] = (name,feats)
    return face_db

def show_result(jdata,frame):
    img_h = frame.shape[0]
    img_w = frame.shape[1]   
    if 'faceList' not in jdata.keys():
        return frame
        
    faceList = jdata['faceList']
    for face in faceList: 
        x = int(face['x'])
        y = int(face['y'])
        w = int(face['w'])
        h = int(face['h'])
        (left, top ,right, bottom) = getValidRect(img_w,img_h, x,y,w,h)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
        if int(face['celeb_id']) > 0:
            user = face['user']
            dist = face['dist']
            info = '%s  %s'%(user,dist)
            frame = cv2ImgAddText(frame, info, left, top - 20, (0, 255, 0), 20)
    return frame


#1. 视频中人脸检测与特征提取
def process(video_path,face_db):
    time_interval = 3
    th = 0.6
    res = process_video(video_path,face_db,PrintNotifier(),time_interval,th)
    #print(res)
    video_result = json.loads(res)
    video_result = sorted(video_result, key=lambda d:d['timestamp'])
    
    cap = cv2.VideoCapture(video_path)
    num = -1
    decResult = {}
    interval =  int(cap.get(cv2.CAP_PROP_FPS)) *  time_interval  
    while(cap.isOpened()):  
        ret, frame = cap.read() 
        num += 1        
        if (ret):
            if (num % interval == 0): 
                i= int(num / interval)      
                frame = show_result(video_result[i],frame)            
                cv2.imshow('image', frame)               
        else:
            break
        key = cv2.waitKey(1)
        if (key & 0xFF) == ord('q'):
            break
        if(key & 0xFF) == ord(' '):
            cv2.waitKey() 
    cap.release()   
    cv2.destroyAllWindows()
    


if __name__ == '__main__': 
    #video_path = 'E:/Data/other/hd_dy_twzd01_20110825.ts'
    video_path = '/home/ysten/yzg/svap/test2.mp4'
    face_db = get_celeba_db() 
    initial()    
    process(video_path,face_db)
    destroy()
    
    '''pkl_file = open('G:/CelebRec/data/celeb_feats.pkl', 'rb')
    pkl_file2 = open('G:/CelebRec/data/celeb_info.pkl', 'wb')
    data1 = pickle.load(pkl_file)
    for d in data1.keys():
         print(d,data1[d])
         break'''
         
         
         
         
         
         
         
         
         
         
         

    
    
    