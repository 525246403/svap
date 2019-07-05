#encoding:utf-8
from __future__ import division
import os

import time

from svap.settings import upload_url, IMAGE_ROOT, MEDIA_ROOT
from svap import settings
import requests
import cv2
from video import models
from datetime import datetime

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'bmp'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def extend_image(img, x, y, w, h):
    img_h, img_w = img.shape[:2]
    ratio = 0.5
    extend_x0 = int(x - ratio * w)
    extend_y0 = int(y - ratio * h)
    extend_x1 = int(x + (1 + ratio) * w)
    extend_y1 = int(y + (1 + ratio) * h)
    if extend_x0 < 0:
        extend_x0 = 0
    if extend_y0 < 0:
        extend_y0 = 0
    if extend_x1 > img_w:
        extend_x1 = img_w
    if extend_y1 > img_h:
        extend_y1 = img_h

    crop_img = img[extend_y0:extend_y1, extend_x0:extend_x1]
    return crop_img

def download(url, media_id):
    media_query = models.Media.objects.filter(media_id=media_id, is_show=True)
    myFile = url.split('/')[-1]
    path = os.path.join(settings.MEDIA_ROOT, myFile)
    response = requests.get(url, stream=True)
    chunk_size = 1024
    content_size = int(response.headers['content-length'])
    size = 0
    if response.status_code == 200:
        with open(path, 'wb') as f:
            for data in response.iter_content(chunk_size=chunk_size):
                f.write(data)
                size += len(data)
                progress = size/content_size
                if progress>=0.990:
                    progress = 0.990
                progress = '%.1f%%' % (progress*100)
                media_query.update(progress=progress)

    if size == content_size:
        new_filename = myFile.split('.')[0] + '_.mp4'
        new_path = os.path.join(settings.MEDIA_ROOT, new_filename)
        os.system(' ffmpeg -i %s -vcodec copy -acodec copy -absf aac_adtstoasc %s' %(path, new_path) )
        final_filename = myFile.split('.')[0] + '.mp4'
        final_path = os.path.join(settings.MEDIA_ROOT, final_filename)
        os.system('qt-faststart %s %s' %(new_path, final_path))

        video_format = final_filename.split('.')[-1]
        file_byte = os.path.getsize(final_path)
        M = 1024 ** 2
        media_size = str(int(file_byte / M)) + 'M'

        time_str = ''
        cap = cv2.VideoCapture(final_path)
        if cap.isOpened():
            rate = cap.get(5)  # 帧速率
            FrameNumber = cap.get(7)  # 视频文件的帧数
            duration = FrameNumber / rate   # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
            M, H = 60, 60 ** 2
            hour = int(duration / H)
            mine = '%02d' % (int(duration % H / M))
            second = '%02d' % (int(duration % H % M))
            time_str = '%s:%s:%s' %(hour, mine, second)

        mediaquery = media_query.first()
        media_url = upload_url + 'media/' + final_filename
        file_url = media_url.split(upload_url)[-1]
        mediaquery.file_url = file_url
        mediaquery.video_format = video_format
        mediaquery.size = media_size
        mediaquery.video_length = time_str
        media_path = final_path.split(MEDIA_ROOT)[-1]
        mediaquery.file_path = media_path
        mediaquery.media_status = '上传成功'
        mediaquery.progress = '100.0%'
        mediaquery.save()
        time.sleep(2)

        count_query = getcountquery()
        if count_query:
            media_count = count_query.first().media_count
            count_query.update(media_count=media_count + 1)
        time.sleep(10)
        os.remove(path)
        os.remove(new_path)

def upload(file, mediaquery):
    videoname = file.name
    path = os.path.join(settings.MEDIA_ROOT, videoname)
    with open(path, 'wb') as f:
        for chunk in file.chunks():
            f.write(chunk)

    new_filename = videoname.split('.')[0] + '_.mp4'
    new_path = os.path.join(settings.MEDIA_ROOT, new_filename)
    os.system('ffmpeg -i %s -vcodec copy -acodec copy -absf aac_adtstoasc %s' % (path, new_path))
    os.remove(path)
    final_filename = videoname.split('.')[0] + '.mp4'
    final_path = os.path.join(settings.MEDIA_ROOT, final_filename)
    os.system('qt-faststart %s %s' % (new_path, final_path))
    os.remove(new_path)

    video_format = final_filename.split('.')[-1]

    file_byte = os.path.getsize(final_path)
    M = 1024 ** 2
    media_size = str(int(file_byte / M)) + 'M'

    time_str = ''
    cap = cv2.VideoCapture(final_path)
    if cap.isOpened():
        rate = cap.get(5)  # 帧速率
        FrameNumber = cap.get(7)  # 视频文件的帧数
        duration = FrameNumber / rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
        M, H = 60, 60 ** 2
        hour = int(duration / H)
        mine = '%02d' % (int(duration % H / M))
        second = '%02d' % (int(duration % H % M))
        time_str = '%s:%s:%s' % (hour, mine, second)

    media_url = upload_url + 'media/' + final_filename
    file_url = media_url.split(upload_url)[-1]
    mediaquery.file_url = file_url
    mediaquery.video_format = video_format
    mediaquery.size = media_size
    mediaquery.video_length = time_str
    media_path = final_path.split(MEDIA_ROOT)[-1]
    mediaquery.file_path = media_path
    mediaquery.media_status = '上传成功'
    mediaquery.save()
    count_query = getcountquery()
    if count_query:
        media_count = count_query.first().media_count
        count_query.update(media_count=media_count + 1)

def getmediadata(mediaquery):
    media_list = []
    for media in mediaquery:
        media_dict = {}
        media_dict['media_id'] = media.media_id
        media_dict['external_id'] = media.external_id
        media_dict['external_system'] = media.external_system
        media_dict['name'] = media.name
        media_dict['video_format'] = media.video_format
        media_dict['size'] = media.size
        media_url = upload_url + media.file_url
        media_dict['file_url'] = media_url
        media_dict['video_length'] = media.video_length
        media_path = MEDIA_ROOT + media.file_path
        media_dict['file_path'] = media_path
        media_dict['media_status'] = media.media_status
        media_dict['progress'] = media.progress
        media_dict['is_keyframe'] = media.is_keyframe
        media_dict['keyframe_progress'] = media.keyframe_progress
        media_dict['keyframe_status'] = media.get_keyframe_status_display()
        media_list.append(media_dict)
    return media_list


def getcountquery():
    t = time.localtime()
    today_date_str = ('%d-%02d-%02d' % (t.tm_year, t.tm_mon, t.tm_mday))
    today_date = datetime.strptime(today_date_str, "%Y-%m-%d")
    count_query = models.Count.objects.filter(create_time=today_date)
    if not count_query:
        famous_count = models.Famous.objects.filter(is_show=True).count()
        face_count = models.Face.objects.filter(is_show=True).count()
        media_count = models.Media.objects.filter(is_show=True).count()
        task_count = models.Task.objects.filter(is_show=True).count()
        models.Count.objects.create(famous_count=famous_count, face_count=face_count, media_count=media_count,
                                    task_count=task_count)
    return count_query

HOST_URL = "http://112.25.72.58:"
detection_task_add_url = HOST_URL  + '8082/detectiontaskadd'
keyframe_task_add_url = HOST_URL  + '8083/keyframetaskadd'


def keyframe_task_add(msg):
    r = requests.post(keyframe_task_add_url, data=msg)
    message = r.text
    return message


def detection_task_add(msg):
    r = requests.post(detection_task_add_url, data=msg)
    message = r.text
    return message


