
import os
import timeit
from glob import glob
import json
import cv2
import numpy as np

from utils import *


@clock
class VideoExtractor(object):
    def __init__(self, total_count, best_count, record_score=False):
        self.total_count = total_count
        self.best_count = best_count
        self.group_count = best_count * 2
        self.group_size = self.total_count // self.group_count
        self.group_index = 0
        self.group_images = []
        self.frame_num = 0
        self.best_images = []
        self.all_scores = []

    def calc_score(self, image):
        score = best_score(image)
        return score

    def add_image(self, image, image_info):
        """
        :param image:
        :param image_info:a dictionary of image information.It must contain "tag"(frame index or file name)
        :return:
        """
        if self.frame_num >= self.total_count:
            raise AssertionError('It only need %d images,but got more.' % self.total_count)
        score = self.calc_score(image)
        image_info['score'] = score
        self.group_index = self.frame_num // self.group_size
        if len(self.group_images) <= self.group_index:  # a new group
            self.group_images.append([image, image_info])
        else:
            if score > self.group_images[self.group_index][1]['score']:
                self.group_images[self.group_index] = [image, image_info]

        self.all_scores.append(image_info)
        self.frame_num += 1
        return self.group_images

    def get_best(self):
        image_list = sorted(self.group_images, key=lambda item: item[1]['score'], reverse=True)
        self.best_images = image_list[:self.best_count]
        return self.best_images

    def get_scores(self):
        return self.all_scores


def test_image(src_root, dst_root, ext_name, best_count=5, record_score=False):
    assert ext_name in ['.png', '.jpg', '.bmp']
    cv2.namedWindow('image')

    files = list_files(src_root, ext_name)
    length = len(files)
    extractor = VideoExtractor(length, best_count)
    for i, f in enumerate(files):
        fn = os.path.join(src_root, f)
        frame = cv2.imread(fn, cv2.IMREAD_COLOR)
        info = {'fn': f}
        extractor.add_image(frame, info)
        cv2.imshow('image', frame)
        cv2.waitKeyEx(1)

    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    best_list = extractor.get_best()
    for best in best_list:
        dst_path = os.path.join(dst_root, best[1]['fn'])
        cv2.imwrite(dst_path, best[0])

    best_images, best_infos = zip(*best_list)
    with open(os.path.join(dst_root, 'best_images.txt'), 'w') as f:
        json.dump(best_infos, f, indent=1)

    score_list = extractor.get_scores()
    if record_score:
        with open(os.path.join(dst_root, 'score_list.txt'), 'w') as f:
            json.dump(score_list, f, indent=1)

    print('successfully extract best image.')


def test_video(src_root, dst_root, file_name, best_count=5, record_score=False):
    cap = cv2.VideoCapture(os.path.join(src_root, file_name))
    ret = cap.isOpened()
    if not ret:
        raise IOError('unable to open video')

    cv2.namedWindow('video')
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('the length is %d' % length)
    extractor = VideoExtractor(length, best_count)
    for i in range(length):
        index = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # stamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        # assert i == index
        ret, frame = cap.read()
        if not ret:
            print('failed to read frame %d from video.' % i)
            continue
            # raise IOError('failed to read frame %d from video.' % i)
        info = {'index': index}
        extractor.add_image(frame, info)
        cv2.imshow('video', frame)
        cv2.waitKeyEx(1)

    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    best_list = extractor.get_best()
    for i, best in enumerate(best_list):
        fn = 'frame%06d_b%d.png' % (best[1]['index'], i)
        dst_path = os.path.join(dst_root, fn)
        cv2.imwrite(dst_path, best[0])

    best_images, best_infos = zip(*best_list)
    with open(os.path.join(dst_root, 'best_images.txt'), 'w') as f:
        json.dump(best_infos, f, indent=1)

    score_list = extractor.get_scores()
    if record_score:
        with open(os.path.join(dst_root, 'score_list.txt'), 'w') as f:
            json.dump(score_list, f, indent=1)

    print('successfully extract best image.')


if __name__ == '__main__':
    src_root = 'I:/Data/video_celebrity/test'
    dst_root = 'I:/Data/video_celebrity/best'

    real_dir = '2bfc67e829e643479d91d1d8e1e7df7d'
    # real_dir = 'tttt'
    extension_name = '.png'
    test_image(os.path.join(src_root, real_dir),
               os.path.join(dst_root, real_dir),
               extension_name, 3, True)

    # real_dir = '2bfc67e829e643479d91d1d8e1e7df7d'
    # file_name = '2bfc67e829e643479d91d1d8e1e7df7d.ts'
    # # real_dir = 'HD1Ma0201c92e6224b1c9490cfb4c3ec6c6c'
    # # file_name = 'HD1Ma0201c92e6224b1c9490cfb4c3ec6c6c.ts'
    # # file_name = 'HD1Ma0201c92e6224b1c9490cfb4c3ec6c6c.ts_20190624_144424.wmv'
    # test_video(os.path.join(src_root, real_dir),
    #            os.path.join(dst_root, real_dir),
    #            file_name)
