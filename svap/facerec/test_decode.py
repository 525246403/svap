# -*- coding: utf-8 -*-
import os
import sys
import json
import cv2
from tqdm import tqdm
import random


def print_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames/fps
    print('Video    : %s' % video_path)
    print('Size     : %dx%d' % (width, height))
    print('Frame    : %d' % total_frames)
    print('FPS      : %d' % fps)
    print('Duration : %.0f' % duration)
    cap.release()
    
    
def decode_video(video_path):
    print_video(video_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    bar = tqdm(total = total_frames)
    while(cap.isOpened()):  
        ret, frame = cap.read() 
        bar.update(1)
        if not ret:
            break
        continue
        key = cv2.waitKey(1)
        if (key & 0xFF) == ord('q'):
            break
        if(key & 0xFF) == ord(' '):
            cv2.waitKey()   
    bar.close()
    cap.release()   
    cv2.destroyAllWindows()
    
def decode_video_rand(video_path, nsegs=-1):
    print_video(video_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_secs = int(total_frames/fps)
    fids = [i*fps for i in range(n_secs)]
    if nsegs < 0:
        nsegs = n_secs
        
    random.shuffle(fids)
    nsegs = min(nsegs, n_secs)
    segs = fids[:nsegs]
    segs.sort()
    print(segs)
    bar = tqdm(total = nsegs)
    for id in segs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, id)
        ret, frame = cap.read()
        bar.update(1)
        if not ret:
            break
        continue
        key = cv2.waitKey(1)
        if (key & 0xFF) == ord('q'):
            break
        if(key & 0xFF) == ord(' '):
            cv2.waitKey()   
    bar.close()
    cap.release()   
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':   
    decode_video('F:/data/start-movies/angelababy/hd_dy_tj_20121213.ts')
    #decode_video_rand('F:/data/start-movies/angelababy/hd_dy_tj_20121213.ts', 50)
    