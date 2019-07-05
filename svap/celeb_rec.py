# -*- coding: utf-8 -*-
import os
import sys
import json
import cv2
import pickle
import base64
from util import getValidRect,extend_image,box2rect,rect2square
from facerec_util import compare_with_celeb
sys.path.append("./facerec")
sys.path.append("./extract_image")
from face_rec import *
from extract_image import VideoExtractor

__use_prefetch = True
__debug = False

IMAGE_SIZE = 128


def get_cache_path(video_path,time_interval):
    cacheDir = os.path.join(os.getcwd(),'videoCache') 
    if not os.path.exists(cacheDir):
        os.makedirs(cacheDir)
    fullname = os.path.basename(video_path)
    name,_ = os.path.splitext(fullname)    
    res_dir = os.path.join(cacheDir,name + '_' + str(time_interval))
    return res_dir
        
class CelebCache(object):
    def __init__(self, video_path, time_interval):
        self.video_path = video_path
        self.time_interval = time_interval         
        self.res_dir = get_cache_path(video_path,time_interval)
        self.res_path = self.res_dir + '.pkl'       
        self.face_dir = os.path.join(self.res_dir,'face')
        self.key_frame_dir = os.path.join(self.res_dir,'key_frame')    
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)   
        if not os.path.exists(self.face_dir):
            os.makedirs(self.face_dir)
        if not os.path.exists(self.key_frame_dir):
            os.makedirs(self.key_frame_dir)                             
            
    def get_video_path(self):
        return self.video_path

    def get_time_interval(self):
        return self.time_interval 
        
    def get_key_frame_dir(self):
        return self.key_frame_dir 
        
    def get_face_dir(self):
        return self.face_dir        
            
    def save_face_image(self,frame,frame_index,face_list):    
        #save image    
        for face in face_list: 
            (x,y,w,h) = box2rect(face['box'])
            image_name = '%d_(%s,%s,%s,%s).jpg'%(frame_index,x,y,w,h)
            image_path = os.path.join(self.face_dir,image_name)
            #rect to square
            (sx,sy,sw,sh) = rect2square(x,y,w,h)
            crop_img = extend_image(frame,sx,sy,sw,sh,0.25)
            if crop_img.shape[0] > IMAGE_SIZE:
                crop_img = cv2.resize(crop_img,(IMAGE_SIZE,IMAGE_SIZE))
            #cv2.imwrite(image_path,crop_img)
            cv2.imencode('.jpg', crop_img)[1].tofile(image_path) #maybe chinese path
            face['path'] = image_path
        return face_list
        
    def save_key_frame(self,best_list):      
        for best in best_list:
            image_name = str(best[1]['index']) + '.jpg'
            dst_path = os.path.join(self.key_frame_dir, image_name)
            print(dst_path)
            #cv2.imwrite(dst_path, best[0])
            cv2.imencode('.jpg', best[0])[1].tofile(dst_path)
      
    #face rec cache      
    def get_rec_cache(self):        
        has_res = False
        decCache = {}
        if os.path.exists(self.res_path) and  os.path.getsize(self.res_path):
            has_res = True
            pkl_file = open(self.res_path, 'rb')
            decCache = pickle.load(pkl_file) 
        return (has_res,decCache)        

    def save_rec_cache(self,decResult):
        fw = open(self.res_path,'wb') 
        pickle.dump(decResult, fw) 
        fw.close() 
        
    #key frame cache      
    def get_keyframe_cache(self):        
        has_res = False
        if os.path.exists(self.key_frame_dir):
            files = os.listdir(self.key_frame_dir)
            if(len(files) > 1) :
               has_res = True 
        return (has_res,self.key_frame_dir)           
    

   
def initial():
    return init()
    
def destroy():
    return close()    
   
def celeb_rec(face_list,face_db,face_th):
    jdata={}  
    faces = []    
    for face in face_list: 
        ret_face = {}
        (x,y,w,h) = box2rect(face['box'])     
        #facerec
        if face_db :           
            fa = face['feat']
            dists = {}
            for i in face_db:
                dist = compare_with_celeb(fa, face_db[i][1])  
                dists[i] =  (face_db[i][0],dist)          
            id_,(user_,dist_) = min(dists.items(), key=lambda d:d[1][1])
            if(dist_ < face_th):#识别出来
                ret_face['celeb_id'] = id_
                ret_face['dist'] = '%.3f' %dist_
                ret_face['user'] = user_
                ret_face['x'] = x
                ret_face['y'] = y
                ret_face['w'] = w
                ret_face['h'] = h 
                ret_face['path'] = face['path']
                faces.append(ret_face)
    if len(faces) > 0:
        jdata['faceList'] = faces
    return jdata

 
#facerec for image
def process_image(image_data):
    jdata = {}     
    image =  cv2.imdecode(np.fromstring(image_data, np.uint8), cv2.IMREAD_COLOR)
    #image = cv2.imread(image_path)
    bbox, points = detect_face(image)
    faces = extract_face_embedding(image, bbox, points)
    jdata['width'] = image.shape[1]
    jdata['height'] = image.shape[0] 
    faceList = []
    for face in faces:
        rface = {}
        (x,y,w,h) = box2rect(face['box'])
        rface['x'] = x
        rface['y'] = y
        rface['w'] = w
        rface['h'] = h
        rface['feat'] = base64.b64encode(face['feat']).decode("utf-8")        
        faceList.append(rface)
    if len(faceList) > 0:
        jdata['faceList'] = faceList
    #print(jdata)
    return  json.dumps(jdata)       
   

def rec_from_cache(decCache,face_db, notifier, face_th):
    rec_result = []
    reporter = SmartReporter(len(decCache.keys()), notifier, itv_pgr=0.5)
    for num in decCache.keys():
        (timestamp,w,h,face_list) = decCache[num]
        jdata = celeb_rec(face_list,face_db,face_th)
        jdata['timestamp'] = timestamp
        jdata['width'] = w
        jdata['height'] = h 
        rec_result.append(jdata)
        reporter(1)
    return rec_result 
 
def rec_from_video(celebCache, face_db, notifier, face_th, best_count,isFaceRec = True,isExtKey = True):
    global __use_prefetch, __debug
    cap = cv2.VideoCapture(celebCache.get_video_path())
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # time to frame
    frame_step = celebCache.get_time_interval() * fps   
    print('Step     : %d' % frame_step)
    print('Pretech  : {0}'.format(__use_prefetch))
    if __use_prefetch:
        prefetch = PrefetchIter(VideoIterator(cap), frame_step)
    else:
        prefetch = BlockIter(VideoIterator(cap), frame_step)
    bar = tqdm(total = prefetch.batch_per_epoch)
    # control notify freqency 5%
    reporter = SmartReporter(prefetch.batch_per_epoch, notifier, itv_pgr=0.05)   
    extractor = VideoExtractor(prefetch.batch_per_epoch, best_count)
    
    # stats
    frame_cnt = -1
    total_done = 0
    total_rec_time = 0
    total_ext_time = 0
    total_faces = 0
    rec_result = []  
    def _report():
        # report stats
        print('')
        print('Step     : %d' % frame_step)
        print('Processed: %d' % total_done)
        if(isFaceRec):
            print('FacRec   : %.3f' % total_rec_time)
            print('FaceNum  : %d' % total_faces)
            print('FacRec * : %.3f' % (total_rec_time/total_done))
        print('ExtKey  : %.3f' % total_ext_time)           
    decResult = {}
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
        
        # face rec
        if(isFaceRec):
            t0 = time.time()
            bbox, points = detect_face(img)
            face_list = extract_face_embedding(img, bbox, points)
            t1 = time.time()
            total_rec_time += t1 - t0
            total_done += 1        
            total_faces += len(face_list)
            
            decResult[frame_index]=(frame_stamp,img.shape[1],img.shape[0],face_list)
            face_list = celebCache.save_face_image(img,frame_index,face_list)                   
            jdata = celeb_rec(face_list,face_db,face_th)
            jdata['timestamp'] = frame_stamp
            jdata['width'] = img.shape[1]
            jdata['height'] = img.shape[0] 
            rec_result.append(jdata)
        
        # extract key frame
        if(isExtKey):
            info = {'index': frame_index}
            t2 = time.time()
            extractor.add_image(frame, info)
            t3 = time.time()
            total_ext_time += t3 - t2        
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
 
    if(isFaceRec) and len(decResult.keys()) > 0:
        celebCache.save_rec_cache(decResult)
        
    if(isExtKey):
        best_list = extractor.get_best()
        celebCache.save_key_frame(best_list)
    # report stats       
    _report()
    cap.release() 
    return rec_result
 
#facerec for video
def process_video(video_path, face_db, notifier, time_interval=1, face_th=0.36, best_count=5):
    print_video(video_path)
    #get cache 
    celebCache = CelebCache(video_path,time_interval)          
    (has_cache,decCache) = celebCache.get_rec_cache()   
    #has cache  
    video_result = {}    
    if has_cache:
        rec_result = rec_from_cache(decCache,face_db, notifier, face_th)           
    else:
        rec_result = rec_from_video(celebCache,face_db,notifier, face_th, best_count) 
        
    video_result['facerec'] = rec_result      
    video_result['keyframe'] = celebCache.get_key_frame_dir()                 
    return json.dumps(video_result)
    

def extract_image(video_path, notifier, time_interval=1, best_count = 5):
    celebCache = CelebCache(video_path,time_interval)
    (has_cache,kCache) = celebCache.get_keyframe_cache()
    if has_cache:
        print('has key frame cache.')
        return kCache
    
    rec_from_video(celebCache,None,notifier,0.0, best_count,isFaceRec = False) 
    return celebCache.get_key_frame_dir()
    
    
def test_img(img_path):
    img = cv2.imread('./facerec/hd_dy_tj_201212[00_10_24][20190612-164120-0].JPG')
    bbox, points = detect_face(img)  
    if bbox is not None:
        face_list = extract_face_embedding(img, bbox, points)
        draw_face_list(img, face_list)            
        cv2.imshow('img', img)
        cv2.waitKey()
        
if __name__ == '__main__':
    initial()
    img_path = './facerec/hd_dy_tj_201212[00_10_24][20190612-164120-0].JPG'    
    res = process_image(img_path) 
    print(res)
    destroy()