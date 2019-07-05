# -*- coding: utf-8 -*-
from __future__ import division
import time

class Speedometer(object):
    def __init__(self):
        self.total = 0
        self.start = time.time()
        self.last = -1
        self.ema = -1

    def __call__(self, count):
        """Callback to Show speed."""
        now = time.time()
        if self.last < 0:
            speed = count/(now - self.start)
            self.ema = speed
            self.last = now
            self.total += count
            return self.ema
        # ema
        speed = count/(now - self.last)
        self.ema = (self.ema * 9 + speed) / 10
        self.last = now
        self.total += count
        return self.ema
    
    @property
    def speed(self):
        return self.ema
        
    @property   
    def avg(self):
        now = time.time()
        return self.total/(now - self.start)


def trim_float(x, n=3):
    fmt = '%%.%df' % n
    x = float(fmt % x)
    return x

    
class SmartReporter(object):
    def __init__(self, total, notifier=None, itv_sec=10, itv_pgr=0.01):
        # speedometer
        self.done = 0
        self.total = total
        self.start = time.time()
        self.last = -1
        # notify
        self.notifier = notifier
        self.itv_sec = itv_sec
        self.itv_pgr = itv_pgr
        # last
        self.last_sec = self.start
        self.last_pgr = 0
        
    def __call__(self, count):
        """Callback to Show speed."""
        epsion = 1e-3
        now = time.time()
        if self.last < 0:
            speed = count/(now - self.start+epsion)
            self.ema = speed
            self.last = now
            self.done += count
        else:
            # ema
            speed = count/(now - self.last+epsion)
            self.ema = (self.ema * 9 + speed) / 10
            self.last = now
            self.done += count
        if self.notifier is None:
            return self.ema
        cur_pgr = self.done/self.total 
        
        # need to report now ?
        if (now - self.last_sec) >= self.itv_sec or (cur_pgr - self.last_pgr) >= self.itv_pgr:
            info = {'total': self.total, 'done': self.done, 'progress': trim_float(cur_pgr)}
            info['speed'] = trim_float(self.ema)
            info['etd'] = trim_float(now - self.start)
            info['eta'] = trim_float((now - self.start) * (1/(epsion+cur_pgr) - 1))
            self.notifier(info)
            # mark
            self.last_pgr = cur_pgr
            self.last_sec = now
        return self.ema
    
    @property
    def speed(self):
        return self.ema
        
    @property   
    def avg(self):
        now = time.time()
        return self.done/(now - self.start+1e-3)
        
    @property
    def progress(self):
        return self.done/self.total
        
        
class PrintNotifier(object):
    def __call__(self, info):
        print(info)
