# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.insert(0, './cv2')

import cv2
print('cv2 ver: ' + cv2.__version__)
import argparse
import numpy as np
import mxnet as mx
import random
import uuid
import time
from tqdm import tqdm

from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
from mxnet_extractor import MxnetExtractor
from zutil.opencv import *
from test_decode import print_video
from alignment import align_to_112x112
from prefetcher import VideoIterator, PrefetchIter, BlockIter
from speedmeter import SmartReporter, PrintNotifier

__detector  = None
__extractor = None
__det_scale = 1
__cache_dir = './cache'   # video meta cache dir
__use_prefetch = True
__debug = False


def cache_path_prefix(url):
    global __cache_dir
    ext = os.path.splitext(url)[1]
    _url = url.encode("utf8")
    try:
        filename = str(uuid.uuid3(uuid.NAMESPACE_DNS, _url)) + ext
    except:
        filename = str(uuid.uuid3(uuid.NAMESPACE_DNS, url)) + ext
    filepath = os.path.join(__cache_dir, filename)
    return filepath


def trim_float(x, n=3):
    fmt = '%%.%df' % n
    x = float(fmt % x)
    return x
    

def init(args=None):
    global __detector, __det_scale, __extractor
    if args is None:
        args = edict()
    # ctx
    ctx = []
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd)>0:
      for i in range(len(cvd.split(','))):
        ctx.append(mx.gpu(i))
    if len(ctx)==0:
      ctx = [mx.cpu()]
      print('use cpu')
    else:
      print('gpu num:', len(ctx))
      
    args.ctx = ctx
    # detector param
    args.det_minsize = 80 * __det_scale
    args.det_threshold = [0.7,0.7,0.8]  # default [0.6,0.7,0.8]
    args.det_factor = 0.708
    args.accurate_landmark = True
    args.mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    detector = MtcnnDetector(model_folder=args.mtcnn_path,
                             minsize=args.det_minsize,
                             factor=args.det_factor,
                             threshold=args.det_threshold,
                             ctx=args.ctx, 
                             num_worker=1, 
                             accurate_landmark = args.accurate_landmark)
    print('init detector done.')
    
    # face embeding model
    args.image_size = [112, 112]
    args.model_path = os.path.join(os.path.dirname(__file__), 'mobileface-zq/mxnet/model')
    extractor = MxnetExtractor(args.model_path, 1, args.image_size, 'fc1_output', ctx=args.ctx)
    print('init face model done.')
    __detector = detector
    __extractor = extractor
    
    
def detect_face(img):
    global __detector
    return __detect_face(__detector, img)

    
def extract_face_embedding(img, bbox, points):
    global __extractor
    return __extract_face_embedding(__extractor, img, bbox, points)


def close():
    global __detector, __extractor
    del __detector
    del __extractor
    __detector  = None
    __extractor = None
    
    
def __detect_face(detector, img):
    global __det_scale
    if __det_scale != 1:
        h, w, c = img.shape
        half_w = int(w*__det_scale)
        half_h = int(h*__det_scale)
        _img = cv2.resize(img, (half_w, half_h))
    else:
        _img = img
    ret = detector.detect_face(_img)
    if ret is None:
        return None, None
    bbox, points = ret
    if bbox.shape[0]==0:
        return None, None
    if __det_scale != 1:
        inv_scale = float(h)/half_h
        bbox = bbox * inv_scale
        points = points * inv_scale
    return bbox, points

    
def __extract_face_embedding(extractor, img, bbox, points):
    face_list = []
    if bbox is None:
        return face_list
    n_faces = bbox.shape[0]
    img_list = []
    for i in range(n_faces):
        box = bbox[i,0:4]
        kpt = points[i, 0:10].reshape(2,5).T
        # plist = [points[i, j] for j in range(10)]
        plist = points[i, 0:10].tolist()
        aligned = align_to_112x112(img, plist)
        img_list.append(aligned)
        face = {'faceindex':i, 'box': box.astype(np.int).tolist(), 
                'kp5': kpt.astype(np.int).tolist(), 'aligned': aligned }
        face_list.append(face)
    face_batch = np.stack(img_list, axis=0)
    feats = extractor.extract(face_batch)
    # split
    for i in range(n_faces):
        feat = feats[i, :]
        face_list[i]['feat'] = feat
    return face_list

def draw_rect(img, box, color=COLOR_GREEN, thickness=1):
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness) 

    
def draw_landmark(img, pt, sz, color):
    for i in range(5):
        cv2.circle(img, (pt[i][0], pt[i][1]), sz, color, -1)

def draw_face_list(img, face_list):
    for face in face_list:
        # show
        draw_rect(img, face['box'], COLOR_RED, 3)
        draw_landmark(img, face['kp5'], 4, COLOR_GREEN)
        cv2.imshow(str(face['faceindex']), face['aligned'])


def test_img(img_path):
    img = cv2.imread('./hd_dy_tj_201212[00_10_24][20190612-164120-0].JPG')
    # cv2.imshow('img', img)
    bbox, points = detect_face(img)
    
    # test time
    for i in range(1):
        start = time.time()
        bbox, points = detect_face(img)
        end = time.time()
        print('%4d : %.3f' % (i, end - start))
    
    
    extract_face_embedding(img, bbox, points)
    # test time
    t_start = time.time()
    for i in range(1):
        start = time.time()
        extract_face_embedding(img, bbox, points)
        end = time.time()
        print('%4d : %.3f' % (i, end - start))
    t_end = time.time()
    print('per-face : %.3f' % ( (t_end - t_start)/100/2 ))
    
    if bbox is not None:
        face_list = extract_face_embedding(img, bbox, points)
        draw_face_list(img, face_list)
            
        cv2.imshow('img', img)
        cv2.waitKey()

    
def process_video(video_path, face_db, notifier, time_interval=1, face_th=0.36):
    global __use_prefetch, __debug
    print_video(video_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # time to frame
    frame_step = time_interval * fps
    print('Step     : %d' % frame_step)
    print('Pretech  : {0}'.format(__use_prefetch))
    if __use_prefetch:
        prefetch = PrefetchIter(VideoIterator(cap), frame_step)
    else:
        prefetch = BlockIter(VideoIterator(cap), frame_step)
    bar = tqdm(total = prefetch.batch_per_epoch)
    # control notify freqency 5%
    reporter = SmartReporter(prefetch.batch_per_epoch, notifier, itv_pgr=0.05)
    
    # stats
    frame_cnt = -1
    total_done = 0
    total_det_time = 0
    total_ext_time = 0
    total_faces = 0
    def _report():
        # report stats
        print('')
        print('Step     : %d' % frame_step)
        print('Processed: %d' % total_done)
        print('Detect   : %.3f' % total_det_time)
        print('Extract  : %.3f' % total_ext_time)
        print('FaceNum  : %d' % total_faces)
        print('Detect * : %.3f' % (total_det_time/total_done))
        if total_faces > 0:
            print('Extract *: %.3f' % (total_ext_time/total_faces))
    
    
    while(True):
        # fetch one frame
        try:
            frame = prefetch.next()[0]
        except:
            break
        
        if frame is None:
            break
            
        frame_cnt += 1
        # process frame
        frame_index = frame_cnt * frame_step
        frame_stamp = frame_index/fps
        img = frame
        t0 = time.time()
        bbox, points = detect_face(img)
        t1 = time.time()
        face_list = extract_face_embedding(img, bbox, points)
        t2 = time.time()
        # tick time
        total_det_time += t1 - t0
        total_ext_time += t2 - t1
        total_done += 1
        total_faces += len(face_list)
        # stats
        #bar.update(1)
        reporter(1)
        # verbose
        if frame_cnt % 100 == 0:
            _report()
        # visualize
        if __debug:
            draw_face_list(img, face_list)
            cv2.imshow('img', img)
            key = cv2.waitKey(1)
            if key == 27:
                break;
            elif key == 32:
                key = cv2.waitKey()
 
    bar.close()
    
    # report stats
    _report()
    cap.release()  
    
    
if __name__ == '__main__':
    init()
    img_path = './hd_dy_tj_201212[00_10_24][20190612-164120-0].JPG'
    video_path = 'E:/Data/other/hd_dy_tj2_20130220.ts'
    #video_path = './hd_dy_tj2_20130220.ts'
    #test_img(img_path)
    #test_video2(video_path)
    process_video(video_path, None, PrintNotifier())
    #close()
    #cv2.destroyAllWindows()
    