#encoding:utf-8
import signal
from threading import Thread
import os
import numpy as np
import time
import random
from multiprocessing import Process, Queue, Value
import sys
import json

from django.db.models import Q

from facerec_util import decode_feat
from svap import settings
from celeb_rec import process_video, initial, destroy, extract_image
from svap.settings import MEDIA_ROOT
from video import models
from video.utils import download, getcountquery

import logging

logger = logging.getLogger('django')

def threadProc(todo, quit_signal, name):
    print('Thread:%s start' % name)
    while quit_signal != 1:
        try:
            task = todo.get()
            file_url = task[0]
            media_id = task[1]
            download(file_url, media_id)
            start = time.time()
            print('Thread:%s start task:%d' % (name, task))
            task_time = random.random() + 0.1
            time.sleep(task_time)
            end = time.time()
            print('Thread:%s finish task:%d in %.2f seconds' % (name, task, end - start))
        except Exception as e:
            #time.sleep(0.5)
            #print(task)
            print(e)
            sys.exit(0)


class MultiThreadLoader:
    def __init__(self, batch_size, nworkers=1, name=''):
        self.batch_size = batch_size
        # todo list
        self.name = name
        self.maxsize = batch_size
        self.todo = Queue(self.maxsize)
        # create threads
        self.quit_signal = Value('i', 0)
        media_query = models.Media.objects.filter(media_status='上传中', is_show=True).order_by('-create_time').all()
        for media in media_query:
            file_url = media.force_url
            media_id = media.media_id
            self.feed((file_url, media_id))

        self.createThread(nworkers)
 
    def createThread(self, nworkers=1):
        self.threads = []
        #self.db_lock = threading.Lock()
        for i in range(nworkers):
            name = self.name + '/upload/' + str(i)
            t = Thread(target=threadProc, args=(self.todo, self.quit_signal, name), name=name)
            t.start()
            self.threads.append(t)

    def feed(self, task):
        if self.todo.full():
            return False
        self.todo.put(task)
        return True

    def getTodo(self):
        if self.todo.full():
            return False
        else:
            return True
        
    def close(self):
        self.quit_signal = 1
        print("mtloader close")

        for t in self.threads:
            try:
                t.terminate()
                t.process.signal(signal.SIGINT) 
            except:
                pass
        for t in self.threads:
            print(t.is_alive())
 
        self.threads = []


mgr = MultiThreadLoader(100, name='upload')

'''
from mt_loader import mgr
mgr.feed(task)
mgrgetTodo()
'''
