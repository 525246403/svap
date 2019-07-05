# -*- coding: utf-8 -*-
from __future__ import absolute_import
import threading

import numpy as np
import cv2


class VideoIterator:
    def __init__(self, cap):
        self.cap = cap
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cursor = 0
        
    def size(self):
        return self.total
        
    def next(self):
        self.cursor += 1
        if self.cursor > self.total:
            return None
            
        ret, frame = self.cap.read()
        return frame
        
    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cursor = 0
        
        
class PrefetchIter(object):
    def __init__(self, iter, batch_size):
        self.iter = iter
        self._batch_size = batch_size
        self._batch_per_epoch = int((iter.size() + batch_size - 1) / batch_size)
        self.data_ready = threading.Event()
        self.data_taken = threading.Event()
        self.data_taken.set()
        self.batch_index = 0

        self.started = True
        self.current_batch = None
        self.next_batch = [None for i in range(batch_size)]

        def prefetch_func(self):
            """Thread entry"""
            while True:
                self.data_taken.wait()
                if not self.started:
                    break
                # load a batch
                self.next_batch[0] = self.iter.next()
                for i in range(1, self._batch_size):
                    self.iter.next()
                self.data_taken.clear()
                self.data_ready.set()
        self.prefetch_threads = threading.Thread(target=prefetch_func, args=[self]) 
        self.prefetch_threads.setDaemon(True)
        self.prefetch_threads.start()


    @property
    def batch_per_epoch(self):
        return self._batch_per_epoch
        
    @property    
    def batch_size(self):
        return self._batch_size
    
    @property
    def size(self):
        return self.iter.size()

    def __len__(self):
        return self._batch_per_epoch

    def __iter__(self):
        return self
 
    def __next__(self):
        return self.next()

    def verbose(self):
        print('TotalSize :%d' % self.iter.size())
        print('BatchSize :%d' % self._batch_size)
        print('BatchEpoch:%d' % self._batch_per_epoch)

    def __del__(self):
        self.started = False
        self.data_taken.set()
        self.prefetch_threads.join()

    def reset(self):
        self.batch_index = 0
        self.data_ready.wait()
        self.iter.reset()
        self.data_ready.clear()
        self.data_taken.set()

    def iter_next(self):
        self.data_ready.wait()
        if self.next_batch[0] is None:
            for i in self.next_batch:
                assert i is None, "Number of entry mismatches between iterators"
            return False
        else:
            self.current_batch = self.next_batch
            self.next_batch = [None for i in range(self._batch_size)]
            self.data_ready.clear()
            self.data_taken.set()
            return True

    def next(self):
        if self.batch_index >= self._batch_per_epoch:
            raise StopIteration
        self.batch_index += 1
        self.iter_next()
        return self.current_batch


class BlockIter(object):
    def __init__(self, iter, batch_size):
        self.iter = iter
        self._batch_size = batch_size
        self._batch_per_epoch = int((iter.size() + batch_size - 1) / batch_size)
        self.batch_index = 0

        self.started = True
        self.current_batch = None
        self.next_batch = [None for i in range(batch_size)]

    def prefetch_func(self):
        """Thread entry"""
        # load a batch
        self.next_batch[0] = self.iter.next()
        for i in range(1, self._batch_size):
            self.iter.next()

    @property
    def batch_per_epoch(self):
        return self._batch_per_epoch
        
    @property    
    def batch_size(self):
        return self._batch_size
    
    @property
    def size(self):
        return self.iter.size()

    def __len__(self):
        return self._batch_per_epoch

    def __iter__(self):
        return self
 
    def __next__(self):
        return self.next()

    def verbose(self):
        print('TotalSize :%d' % self.iter.size())
        print('BatchSize :%d' % self._batch_size)
        print('BatchEpoch:%d' % self._batch_per_epoch)

    def __del__(self):
        self.started = False
        
    def reset(self):
        self.batch_index = 0
        self.iter.reset()

    def iter_next(self):
        self.prefetch_func()
        if self.next_batch[0] is None:
            for i in self.next_batch:
                assert i is None, "Number of entry mismatches between iterators"
            return False
        else:
            self.current_batch = self.next_batch
            self.next_batch = [None for i in range(self._batch_size)]
            return True

    def next(self):
        if self.batch_index >= self._batch_per_epoch:
            raise StopIteration
        self.batch_index += 1
        self.iter_next()
        return self.current_batch
        