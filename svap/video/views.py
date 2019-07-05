#encoding:utf-8

import time
import json
import io
import os

#import urllib.request, urllib.error
import urllib
import urllib2
import uuid
from datetime import datetime, timedelta

import cv2
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import requests
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.db.models import Q
from django.http import HttpResponse
from django.shortcuts import render

from celeb_rec import process_video, process_image, initial, destroy
from svap import settings
from svap.settings import upload_url, IMAGE_ROOT, MEDIA_ROOT, TASK_RESU_ROOT
from video import models
from video.utils import allowed_file, extend_image,  getmediadata, upload, getcountquery, keyframe_task_add, detection_task_add
from facerec_util import decode_feat
from mt_loader import mgr
import pickle
initial()
import logging
logger = logging.getLogger('django')
# /video/home
def index(request):
    t = time.localtime()
    today_date_str = ('%d-%02d-%02d' % (t.tm_year, t.tm_mon, t.tm_mday))
    today_date = datetime.strptime(today_date_str, "%Y-%m-%d")

    count_query = models.Count.objects.filter(create_time=today_date).first()
    print(count_query)
    if not count_query:
        famous_count = models.Famous.objects.filter(is_show=True).count()
        face_count = models.Face.objects.filter(is_show=True).count()
        media_count = models.Media.objects.filter(is_show=True).count()
        task_count = models.Task.objects.filter(is_show=True).count()
        models.Count.objects.create(famous_count=famous_count, face_count=face_count, media_count=media_count, task_count=task_count)
        time_list = []
        task_count_list = []
        for i in range(0, 31):
            begin_date = today_date - timedelta(days=i)
            begin_date_str = begin_date.strftime('%Y-%m-%d')
            count_query = models.Count.objects.filter(create_time=begin_date_str).first()
            if not count_query:
                day_task_count = 0
            else:
                day_task_count = count_query.task_count
            task_count_list.append(day_task_count)
            time_list.append(begin_date_str)
        task_count_list.reverse()
        time_list.reverse()
        data = {
            'famous_count': famous_count,
            'face_count': face_count,
            'media_count': media_count,
            'task_count': task_count,
            'task_count_list': task_count_list,
            'time_list': time_list,
        }

        return render(request, 'video/home.html',data)

    famous_count = count_query.famous_count
    face_count = count_query.face_count
    media_count = count_query.media_count
    task_count = count_query.task_count
    time_list = []
    task_count_list = []
    for i in range(0, 31):
        begin_date = today_date - timedelta(days=i)
        begin_date_str = begin_date.strftime('%Y-%m-%d')
        count_query = models.Count.objects.filter(create_time=begin_date_str).first()
        if not count_query:
            day_task_count = 0
        else:
            day_task_count = count_query.task_count
        task_count_list.append(day_task_count)
        time_list.append(begin_date_str)
    task_count_list.reverse()
    time_list.reverse()
    data = {
        'famous_count': famous_count,
        'face_count': face_count,
        'media_count': media_count,
        'task_count': task_count,
        'task_count_list': task_count_list,
        'time_list': time_list,
    }
    return render(request, 'video/home.html', data)

# /video/targetrec
def target_recognition(request):
    if request.method == 'POST':
        task_query = models.Task.objects.filter(is_show=True, state=3).first()
        if not task_query:
            return HttpResponse(json.dumps({"ret": 1, "val": 'No ongoing tasks', "error": ""}))
        progress = task_query.progress
        print(progress)
        task_id = task_query.task_id
        data = {
            'progress':progress,
            'task_id':task_id
        }
        return HttpResponse(json.dumps({"ret": 0, "val": data, "error": ""}))

    task_query = models.Task.objects.filter(is_show=True).order_by('-create_time').all()

    paginator = Paginator(task_query, 20)
    page = request.GET.get('page', 1)
    total_page = paginator.num_pages
    try:
        contacts = paginator.page(page)
    except PageNotAnInteger:
        contacts = paginator.page(1)
    except EmptyPage:
        contacts = paginator.page(paginator.num_pages)

    task_list = []
    for contact in contacts:
        task_dict = {}
        task_dict['task_id'] = contact.task_id
        task_dict['task_media_id'] = contact.task_media_id
        task_dict['progress'] = contact.progress
        task_dict['name'] = contact.name
        target_face_id = contact.target_face_id
        target_face_id = json.loads(target_face_id)
        target_face_name_list = []
        for target_face in target_face_id:
            target_face_query = models.Famous.objects.filter(id=int(target_face)).first()
            target_face_name = target_face_query.name
            target_face_name_list.append(target_face_name.encode('utf-8'))
        task_dict['target_face_name_list'] = ','.join(target_face_name_list)
        task_dict['create_time'] = contact.create_time.strftime('%Y-%m-%d %H:%M:%S')
        task_dict['state'] = contact.get_state_display()
        task_dict['op'] = '查看结果' if contact.state==1 else ''
        task_list.append(task_dict)
    data = {
        'task_list':task_list,
        'total_page':total_page if total_page>1 else '',
        'contacts':contacts,
        'page':page
    }
    return render(request, 'video/target_recognition.html', data)

# /video/targetmag
def target_management(request):
    if request.method=='POST':
        famous_id = request.POST.get('famousid')
        famous_name = request.POST.get('famousname')
        famous = models.Famous.objects.filter(id=famous_id, is_show=True).first()
        if not famous:
            return HttpResponse(json.dumps({"ret": 0, "val": "Please enter the correct famous ID", "error": ""}))
        p_type = famous.type
        birthplace = famous.birthplace
        birthday = famous.birthday
        height = famous.height
        university = famous.university
        works = famous.works
        famous_updata_time = famous.updata_time.strftime('%Y-%m-%d %H:%M:%S')

        face_list = models.Face.objects.filter(famous_id=famous_id, is_show=True).order_by('-updata_time').values_list('url', 'id')
        face_url_list = []
        face_id_list = []
        face_length  = 0
        if face_list:
            face_length = len(face_list)
            for url_dict in face_list:
                img_url = upload_url + url_dict[0]
                face_url_list.append(img_url)
                face_id_list.append(url_dict[1])
        request.session['famous_id'] = famous_id
        data = {
            'p_type':p_type,
            'famous_name': famous_name.encode('utf-8'),
            'famous_id': famous_id,
            'face_length': face_length,
            'face_url_list':face_url_list,
            'face_id_list':face_id_list,
            'famous_updata_time':famous_updata_time,
            'birthplace':birthplace,
            'birthday':birthday,
            'height':height,
            'university':university,
            'works':works,
        }
        data = json.dumps(data)
        return HttpResponse(data)
    famous_list = models.Famous.objects.filter(is_show=True, type=1).all()
    politics_list = models.Famous.objects.filter(is_show=True, type=2).all()
    famous_id = request.session.get("famous_id")
    if not famous_id:
        famous_id = models.Famous.objects.filter(is_show=True).first().id
    print(famous_id)
    famous = models.Famous.objects.filter(id=famous_id, is_show=True).first()
    if not famous:
        data = {
            'famous_list': famous_list,
            'politics_list':politics_list,
            'p_type':3,
        }
        return render(request, 'video/target_management.html',data)
    p_type = famous.type
    famous_name = famous.name
    famous_id = famous.id
    birthplace = famous.birthplace
    birthday = famous.birthday
    height = famous.height
    university = famous.university
    works = famous.works
    famous_updata_time = famous.updata_time.strftime('%Y-%m-%d %H:%M:%S')

    face_query = models.Face.objects.filter(famous_id=famous_id, is_show=True).order_by('-updata_time').all()

    paginator = Paginator(face_query, 21)
    page = request.GET.get('page', 1)
    total_page = paginator.num_pages
    try:
        contacts = paginator.page(page)
    except PageNotAnInteger:
        contacts = paginator.page(1)
    except EmptyPage:
        contacts = paginator.page(paginator.num_pages)

    face_list = []
    for contact in contacts:
        face_dict = {}
        img_url= upload_url + contact.url
        face_dict['url'] = img_url
        face_dict['id'] = contact.id
        img_path = IMAGE_ROOT + contact.path
        face_dict['file_path'] = img_path
        face_dict['feat'] = contact.feat
        face_list.append(face_dict)

    face_length = len(face_query)
    data = {
        'famous_name':famous_name.encode('utf-8'),
        'famous_id':famous_id,
        'famous_updata_time':famous_updata_time,
        'face_list':face_list,
        'face_length':face_length,
        'famous_list':famous_list,
        'politics_list':politics_list,
        'total_page':total_page,
        'birthplace':birthplace,
        'birthday':birthday,
        'height':height,
        'university':university,
        'works':works,
        'page':page,
        'p_type':p_type,
    }
    return render(request, 'video/target_management.html', data)

# video/getfacepic
def get_facepic(request):
    if request.method=='POST':
        famous_id = request.POST.get('famous_id')
        face_query = models.Face.objects.filter(famous_id=famous_id, is_show=True).order_by('-updata_time').all()
        paginator = Paginator(face_query, 21)
        page = request.POST.get('page', 1)
        total_page = paginator.num_pages
        try:
            contacts = paginator.page(page)
        except PageNotAnInteger:
            contacts = paginator.page(1)
        except EmptyPage:
            contacts = paginator.page(paginator.num_pages)

        face_list = []
        for contact in contacts:
            face_dict = {}
            img_url = upload_url + contact.url
            face_dict['url'] = img_url
            face_dict['id'] = contact.id
            img_path = IMAGE_ROOT + contact.path
            face_dict['file_path'] = img_path
            face_dict['feat'] = contact.feat
            face_list.append(face_dict)

        data = {
            'famous_id':famous_id,
            'face_list':face_list,
            'total_page':total_page,
            'page':page,
        }
        return HttpResponse(json.dumps(data))

# /video/getfacelist
def get_face_list(request):
    if request.method=='POST':
        files = request.FILES.getlist('file')
        face_list = []
        for file in files:
            if file and allowed_file(file.name):
                data = file.file.read()
                #initial()
                jdata = process_image(data)
                #destroy()
                jdata = json.loads(jdata)
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                facelist = jdata.get('faceList')
                if facelist:
                    for i in facelist:
                        feat = i.get('feat')
                        if feat:
                            w = i.get('w')
                            h = i.get('h')
                            x = i.get('x')
                            y = i.get('y')
                            crop_img = extend_image(img, x, y, w, h)
                            feat = feat.encode()
                            img_name = str(uuid.uuid3(uuid.NAMESPACE_DNS, feat)) + "." + file.name.rsplit('.', 1)[1]

                            file_url =  upload_url + "image/" +'image' + '/' + img_name
                            img_path = os.path.join(settings.IMAGE_ROOT, 'image', img_name)
                            print(img_path)
                            print(file_url)
                            cv2.imwrite(img_path, crop_img)
                            face = {'file_url': file_url, 'file_path': img_path, 'feat':feat}
                            face_list.append(face)
        data = json.dumps(face_list)
        return HttpResponse(data)

# /video/newfamous
def new_famous(request):
    if request.method == 'POST':
        data = request.POST.get('val')
        famous_name = request.POST.get('famousname')
        p_type = request.POST.get('p_type')
        face_list = json.loads(data)
        if not famous_name:
            return HttpResponse(json.dumps({"ret": 1, "val": "Please enter name", "error": ""}))
        if not p_type:
            return HttpResponse(json.dumps({"ret": 1, "val": "Please choose the category", "error": ""}))

        models.Famous.objects.create(name=famous_name, type=int(p_type))
        for face in face_list:
            img_url = face.get('url')
            url = img_url.split(upload_url)[-1]
            img_path = face.get('path')
            path = img_path.split(IMAGE_ROOT)[-1]
            feat = face.get('feat')
            famous_id = models.Famous.objects.filter(name=famous_name).order_by('-id')[0].id
            request.session['famous_id'] = famous_id
            models.Face.objects.create(feat=feat, url=url, path=path, famous_id=famous_id)

        count_query = getcountquery()
        if count_query:
            famous_count = count_query.first().famous_count
            face_count = count_query.first().face_count
            count_query.update(famous_count=famous_count + 1)
            count_query.update(face_count=face_count + len(face_list))
        return HttpResponse(json.dumps({"ret": 0, "val": "success", "error": ""}))

# /video/addface
def add_face(request):
    if request.method == 'POST':
        data = request.POST.get('val')
        face_list = json.loads(data)
        for face in face_list:
            img_url = face.get('url')
            url = img_url.split(upload_url)[-1]
            img_path = face.get('path')
            path = img_path.split(IMAGE_ROOT)[-1]
            feat = face.get('feat')
            famous_id = face.get('famous_id')
            if famous_id:
                models.Face.objects.create(feat=feat, url=url, path=path, famous_id=famous_id)
                updata_time = datetime.now()
                models.Famous.objects.filter(id=famous_id).update(updata_time=updata_time)
                count_query = getcountquery()
                if count_query:
                    face_count = count_query.first().face_count
                    count_query.update(face_count=face_count + 1)
                request.session['famous_id'] = famous_id
        return HttpResponse(json.dumps({"ret": 0, "val": "success", "error": ""}))

# /video/autoaddface
def auto_add_face(request):
    if request.method=='POST':
        files = request.FILES.getlist('file')
        famous_id = request.POST.get('famous_id')
        if not files:
            return HttpResponse(json.dumps({"ret": 1, "val": "please upload file", "error": ""}))
        if not famous_id:
            return HttpResponse(json.dumps({"ret": 1, "val": "please enter famous_id", "error": ""}))
        for file in files:
            if file and allowed_file(file.name):
                data = file.file.read()
                #initial()
                jdata = process_image(data)
                #destroy()
                jdata = json.loads(jdata)
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                facelist = jdata.get('faceList')
                if facelist:
                    for i in facelist:
                        feat = i.get('feat')
                        if feat:
                            w = i.get('w')
                            h = i.get('h')
                            x = i.get('x')
                            y = i.get('y')
                            crop_img = extend_image(img, x, y, w, h)
                            img_name = str(uuid.uuid3(uuid.NAMESPACE_DNS, feat)) + "." + file.name.rsplit('.', 1)[1]
                            file_url = "image/" + 'image' + '/'+ img_name
                            img_path = os.path.join(settings.MEDIA_ROOT, 'image', img_name)
                            cv2.imwrite(img_path, crop_img)
                            path = img_path.split(MEDIA_ROOT)[-1]
                            models.Face.objects.create(feat=feat, url=file_url, path=path, famous_id=famous_id)
                            updata_time = datetime.now()
                            models.Famous.objects.filter(id=famous_id).update(updata_time=updata_time)
                            count_query = getcountquery()
                            if count_query:
                                face_count = count_query.first().face_count
                                count_query.update(face_count=face_count + 1)
        return HttpResponse(json.dumps({"ret": 0, "val": "success", "error": ""}))

# /video/delface
def del_face(request):
    if request.method == 'POST':
        data = request.POST.get('val')
        del_face_list = json.loads(data)
        if del_face_list:
            for face in del_face_list:
                id = int(face.get('id'))
                models.Face.objects.filter(id=id).update(is_show=False)
                count_query = getcountquery()
                if count_query:
                    face_count = count_query.first().face_count
                    count_query.update(face_count=face_count - 1)
        return HttpResponse(json.dumps({"ret": 0, "val": "success", "error": ""}))

# /video/getfacefeat
def get_face_feat(request):
    if request.method == 'POST':
        famous_name = request.POST.get('name')
        famous_query = models.Famous.objects.filter(name=famous_name).values('id')
        famous_dict = {}
        if famous_query:
            for famous in famous_query:
                famous_id = famous.get('id')
                face_query = models.Face.objects.filter(famous_id=famous_id).values_list('feat', 'url')
                feat_list = []
                url_list = []
                if face_query:
                    for face in face_query:
                        feat_list.append(face[0])
                        img_url = upload_url + face[1]
                        url_list.append(img_url)
                famous_dict[famous_id] = (famous_name, feat_list)
        return HttpResponse(json.dumps(famous_dict, ensure_ascii=False))

# /video/showvideo
def showvideo(request):
    task_id = request.GET.get('task_id')
    task_query = models.Task.objects.filter(task_id=task_id).first()
    if not task_query:
        return HttpResponse(json.dumps({"ret": 1, "val": "Please enter the correct task ID", "error": ""}))
    video_name = task_query.name

    resu_file_name = str(task_id) + '.json'
    resu_path = os.path.join(settings.TASK_RESU_ROOT, 'resu', resu_file_name)
    with open(resu_path, 'r') as f:
        resu = json.load(f)
    task_media_id = task_query.task_media_id
    media_query = models.Media.objects.filter(media_id=task_media_id).first()
    video_format = media_query.video_format
    size = media_query.size
    video_length = media_query.video_length
    media_url = media_query.file_url
    file_url = upload_url + media_url
    task_face_id_list = json.loads(task_query.target_face_id)
    task_face_dict = {}
    famous_name_list = []
    for task_face_id in task_face_id_list:
        famous_query = models.Famous.objects.filter(id=int(task_face_id)).first()
        famous_name = famous_query.name
        famous_name_list.append(famous_name)
        task_face_dict[task_face_id.encode('utf-8')] = famous_name.encode('utf-8')
    task_resu_list = json.loads(resu).get('facerec')
    task_resu_ = json.dumps(task_resu_list)
    task_face_str = json.dumps(task_face_dict)
    famous_point_dict = {}
    famous_img_dict = {}
    for famous in famous_name_list:
        famous_point_dict[famous] = []
        famous_img_dict[famous] = []

    for task_resu_dict in task_resu_list:

        famousfaceList = task_resu_dict.get('faceList')
        if famousfaceList:
            for famous_dict in famousfaceList:
                famous = famous_dict.get('user')
                timestamp = task_resu_dict.get('timestamp')
                path_ = famous_dict.get('path')
                if path_:
                    path = path_.split(settings.BASE_DIR)[-1]
                    #path = path_.split(settings.BASE_DIR)[-1].replace('\\', '/')
                    famous_img_dict[famous].append(path)
                famous_point_dict[famous].append(timestamp)

    data = {
        'video_name':video_name,
        'task_resu':task_resu_,
        'video_format':video_format,
        'size':size,
        'video_length':video_length,
        'file_url':file_url,
        'task_face_dict':task_face_str,
        'famous_name_list':famous_name_list,
        'famous_name_str':json.dumps(famous_name_list),
        'famous_point_dict':json.dumps(famous_point_dict),
        'famous_img_dict':json.dumps(famous_img_dict),

    }
    return render(request, 'video/showvideo.html', data)

# /video/famoustask
def famoustask(request):
    famous_list_query = models.Famous.objects.filter(is_show=True, type=1).all()
    famous_list = []
    for famous in famous_list_query:
        famous_dict = {}
        famous_id = famous.id
        famous_name = famous.name
        famous_dict['id'] = famous_id
        famous_dict['name'] = famous_name
        famous_list.append(famous_dict)
    data = {
        'famous_list': famous_list,
    }
    return render(request, 'video/famoustask.html', data)

# /video/politicstask
def politicstask(request):
    famous_list_query = models.Famous.objects.filter(is_show=True, type=2).all()
    famous_list = []
    for famous in famous_list_query:
        famous_dict = {}
        famous_id = famous.id
        famous_name = famous.name
        famous_dict['id'] = famous_id
        famous_dict['name'] = famous_name
        famous_list.append(famous_dict)
    data = {
        'famous_list': famous_list,
    }
    return render(request, 'video/politicstask.html', data)

# /video/deltask
def del_task(request):
    if request.method=='POST':
        task_list = json.loads(request.POST.get('task_list'))
        if not task_list:
            return HttpResponse(json.dumps({"ret": 1, "val": 'no tasks selected', "error": ""}))
        for task_id in task_list:
            id = int(task_id)
            task_query = models.Task.objects.filter(task_id=id, is_show=True)
            if not task_query:
                return HttpResponse(json.dumps({"ret": 1, "val": 'Please enter the correct task ID', "error": ""}))
            task_query.update(is_show=False)
            count_query = getcountquery()
            if count_query:
                task_count = count_query.first().task_count
                count_query.update(task_count=task_count - 1)
        return HttpResponse(json.dumps({"ret": 0, "val": 'OK', "error": ""}))

# /video/videoadmin
def video_admin(request):
    if request.method=='POST':
        media_query = models.Media.objects.filter(Q(is_show=True, keyframe_status=2)|Q(is_show=True, media_status='上传中')).first()
        if not media_query :
            return HttpResponse(json.dumps({"ret": 1, "val": 'No tasks in download', "error": ""}))

        media_id = media_query.media_id
        progress = media_query.progress
        keyframe_progress= media_query.keyframe_progress
        keyframe_status= media_query.get_keyframe_status_display()
        data = {
            'progress': progress,
            'media_id': media_id,
            'keyframe_progress': keyframe_progress,
            'keyframe_status':keyframe_status
        }
        return HttpResponse(json.dumps({"ret": 0, "val": data, "error": ""}))
    media_query = models.Media.objects.filter(is_show=True).all()
    paginator = Paginator(media_query, 20)
    page = request.GET.get('page', 1)
    total_page = paginator.num_pages
    try:
        contacts = paginator.page(page)
    except PageNotAnInteger:
        contacts = paginator.page(1)
    except EmptyPage:
        contacts = paginator.page(paginator.num_pages)
    media_list = []
    for media in contacts:
        media_dict = {}
        media_dict['media_id'] = media.media_id
        media_dict['external_id'] = media.external_id
        media_dict['external_system'] = media.external_system
        media_dict['name'] = media.name
        media_url = upload_url + media.file_url if media.file_url else media.file_url
        media_dict['file_url'] = media_url
        media_dict['video_format'] = media.video_format
        media_dict['size'] = media.size
        media_dict['progress'] = media.progress
        media_dict['video_length'] = media.video_length
        media_path = MEDIA_ROOT + media.file_path if media.file_path else media.file_path
        media_dict['file_path'] = media_path
        media_dict['media_status'] = media.media_status
        media_dict['is_keyframe'] = media.is_keyframe
        media_dict['keyframe_progress'] = media.keyframe_progress
        media_dict['keyframe_status'] = media.get_keyframe_status_display()
        media_list.append(media_dict)
    data = {
        'media_list':media_list,
        'total_page': total_page if total_page > 1 else '',
        'page':page,
        'contacts':contacts
    }
    return render(request, 'video/video_admin.html',data)

# /video/searchvideo
def search_video(request):
    if request.method == "POST":
        videoid =  request.POST.get('videoid')
        videoname =  request.POST.get('videoname')

        if not videoid:
            mediaquery = models.Media.objects.filter(name=videoname).all()
            if not mediaquery:
                return HttpResponse(json.dumps({"ret": 1, "val": "Please enter the video ID", "error": ""}))
            else:
                media_list = getmediadata(mediaquery)
                return HttpResponse(json.dumps({"ret": 0, "val": json.dumps(media_list), "error": ""}))

        if not videoname:
            mediaquery = models.Media.objects.filter(media_id=videoid).all()
            if not mediaquery:
                return HttpResponse(json.dumps({"ret": 1, "val": "Please enter the video name", "error": ""}))
            else:
                media_list = getmediadata(mediaquery)
                print(media_list)
                return HttpResponse(json.dumps({"ret": 0, "val": json.dumps(media_list), "error": ""}))


        mediaquery = models.Media.objects.filter(media_id=videoid, name=videoname).all()
        if not mediaquery:
            return HttpResponse(json.dumps({"ret": 1, "val": "Please enter the correct query conditions", "error": ""}))
        else:
            media_list = getmediadata(mediaquery)
            return HttpResponse(json.dumps({"ret": 0, "val": json.dumps(media_list), "error": ""}))

# /video/delvideo
def del_video(request):
    if request.method == 'POST':
        media_id_str = request.POST.get('val')
        if media_id_str:
            media_id_list = json.loads(media_id_str)
            for media_id in media_id_list:
                media_query = models.Media.objects.filter(media_id=media_id)
                if media_query:
                    media_query.update(is_show=False)
                    count_query = getcountquery()
                    if count_query:
                        media_count = count_query.first().media_count
                        count_query.update(media_count=media_count - 1)
            return HttpResponse(json.dumps({"ret": 0, "val": '', "error": ""}))

# /video/changevideo
def change_video(request):
    if request.method == 'POST':
        media_id = request.POST.get('media_id')
        external_id = request.POST.get('external_id')
        external_system = request.POST.get('external_system')
        media_name = request.POST.get('media_name')

        media_query = models.Media.objects.filter(media_id=media_id)
        if not media_query:
            return HttpResponse(json.dumps({"ret": 1, "val": 'Video id error', "error": ""}))
        media_query.update(external_id=external_id, external_system=external_system, name=media_name)

        return HttpResponse(json.dumps({"ret": 0, "val": '', "error": ""}))

# /video/getvideo
def get_video(request):
    if request.method == 'POST':
        media_id_str = request.POST.get('val')
        if not media_id_str:
            return HttpResponse(json.dumps({"ret": 1, "val": 'Please input media_id', "error": ""}))

        media_query = models.Media.objects.filter(media_id=media_id_str, media_status='上传成功', is_show=True).first()
        if media_query:
            media_id = media_id_str
            external_id = media_query.external_id
            external_system = media_query.external_system
            name = media_query.name
            media_url = upload_url + media_query.file_url
            file_url = media_url
            video_format = media_query.video_format
            size = media_query.size
            video_length = media_query.video_length
            media_path = MEDIA_ROOT + media_query.file_path
            file_path = media_path
            media_status = media_query.media_status
            data = {
                'media_id': media_id,
                'external_id': external_id,
                'external_system': external_system,
                'name': name,
                'file_url': file_url,
                'video_format': video_format,
                'size': size,
                'video_length': video_length,
                'file_path': file_path,
                'media_status': media_status,
            }
            return HttpResponse(json.dumps({"ret": 0, "val": data, "error": ""}))
        else:
            return HttpResponse(json.dumps({"ret": 1, "val": 'Please re-input media_id', "error": ""}))

# /video/getfamous
def get_famous(request):
    if request.method == 'POST':
        famousid = request.POST.get('val')
        if not famousid:
            return HttpResponse(json.dumps({"ret": 2, "val": 'Please input famousname', "error": ""}))
        famous_query = models.Famous.objects.filter(id=famousid, is_show=True).all()
        if not famous_query:
            return HttpResponse(json.dumps({"ret": 1, "val": 'Please re-input famousname', "error": ""}))
        else:
            famous_list = []
            for famous in famous_query:
                famous_dict = {}
                famous_id = famous.id
                famous_name = famous.name
                famous_face_list = models.Face.objects.filter(famous_id=famous_id, is_show=True)
                if famous_face_list:
                    famous_face_url = upload_url + famous_face_list.first().url
                    famous_dict['url'] = famous_face_url
                famous_dict['id'] = famous_id
                famous_dict['name'] = famous_name
                famous_list.append(famous_dict)
            return HttpResponse(json.dumps({"ret": 0, "val": json.dumps(famous_list), "error": ""}))

# /video/changefamous
def change_famous(request):
    if request.method == 'POST':
        work = request.POST.get('work')
        print(work)
        birthday = request.POST.get('birthday')
        birthplace = request.POST.get('birthplace')
        height = request.POST.get('height')
        university = request.POST.get('university')
        famous_id = request.POST.get('famous_id')
        famous_query = models.Famous.objects.filter(id=famous_id)
        if not famous_query:
            return HttpResponse(json.dumps({"ret": 1, "val": 'wrong famous id', "error": ""}))

        famous_query.update(works=work, birthday=birthday, birthplace=birthplace, height=height, university=university)
    return HttpResponse(json.dumps({"ret": 0, "val": 'OK', "error": ""}))

# /video/delfamous
def del_famous(request):
    if request.method == 'POST':
        famous_id = request.POST.get('val')
        if not famous_id:
            return HttpResponse(json.dumps({"ret": 1, "val": 'can not get famous id', "error": ""}))
        famous_query = models.Famous.objects.filter(id=famous_id)
        if not famous_query:
            return HttpResponse(json.dumps({"ret": 1, "val": 'wrong famous id', "error": ""}))
        famous_query.update(is_show=False)

        return HttpResponse(json.dumps({"ret": 0, "val": 'hah', "error": ""}))

# /video/getkeyframeprogress
def key_frame_progress(request):
    if request.method == 'POST':
        data = request.POST.get("data")
        data = json.loads(data)
        media_id = data.get('media_id')
        progress = data.get('progress')
        media_query = models.Media.objects.filter(media_id=media_id)
        media_query.update(keyframe_progress=progress)
        return HttpResponse(json.dumps({"ret": 0, "val": 'ok', "error": ""}))


# /video/getkeyframestate
def key_frame_state(request):
    if request.method == 'POST':
        data = request.POST.get("data")
        data = json.loads(data)
        media_id = data.get('media_id')
        state = data.get('state')
        media_query = models.Media.objects.filter(media_id=media_id)
        if int(state) == 3:
            media_query.update(is_keyframe=True, keyframe_status=state)
        else:
            media_query.update(keyframe_status=state)
        return HttpResponse(json.dumps({"ret": 0, "val": 'ok', "error": ""}))

# /video/keyframetask
def key_frame_task(request):
    if request.method == 'POST':
        media_id_str = request.POST.get('val')
        if media_id_str:
            media_id_list = json.loads(media_id_str)
            for media_id in media_id_list:
                media_query = models.Media.objects.filter(media_id=media_id)
                if not media_query:
                    return HttpResponse(json.dumps({"ret": 1, "val": 'Please select the correct video ID', "error": ""}))
                media_path = MEDIA_ROOT + media_query.first().file_path
                media_query.update(keyframe_status=1)
                #keyframe_task.feed((media_path, media_id))
                jdata = {"media_path":media_path, "media_id":media_id}
                keyframe_task_add(jdata)
        return HttpResponse(json.dumps({"ret": 0, "val": '', "error": ""}))

    media_id = request.GET.get('media_id')
    mediaquery = models.Media.objects.filter(media_id=media_id, is_keyframe=True).first()
    if not mediaquery:
        return HttpResponse(json.dumps({"ret": 1, "val": "Please enter the correct media ID", "error": ""}))

    resu_file_name = str(media_id) + '.txt'
    resu_path = os.path.join(settings.TASK_RESU_ROOT, 'resu', resu_file_name)
    video_name = mediaquery.name
    video_format = mediaquery.video_format
    size = mediaquery.size
    video_length = mediaquery.video_length
    data = {
        'video_name':video_name,
        'video_format':video_format,
        'size':size,
        'video_length':video_length,
    }
    if not os.path.exists(resu_path):
        return render(request, 'video/showvideokeyframe.html', data)
    with open(resu_path, 'r') as f:
        resu = f.read()
    face_list = []
    key_face_dir = resu.split(settings.BASE_DIR)[-1].replace('\\', '/')
    file_list = os.listdir(resu)
    for file in file_list:
        file_path = key_face_dir + '/' + file
        face_list.append(file_path)
    data = {
        'face_list':face_list,
        'video_name': video_name,
        'video_format': video_format,
        'size': size,
        'video_length': video_length,
    }

    return render(request, 'video/showvideokeyframe.html', data)

# /video/getdetectiontaskprogress
def detection_task_progress(request):
    if request.method == 'POST':
        data = request.POST.get("data")
        print(data)
        data = json.loads(data)
        task_id = data.get('task_id')
        progress = data.get('progress')
        task_query = models.Task.objects.filter(task_id=task_id)
        task_query.update(progress=progress)
        return HttpResponse(json.dumps({"ret": 0, "val": 'ok', "error": ""}))


# /video/getdetectiontaskstate
def detection_task_state(request):
    if request.method == 'POST':
        data = request.POST.get("data")
        print(data)
        data = json.loads(data)
        task_id = data.get('task_id')
        state = data.get('state')
        task_query = models.Task.objects.filter(task_id=task_id)
        if int(state) == 1:
            task_query.update(state=state)
        else:
            task_query.update(state=state)
        return HttpResponse(json.dumps({"ret": 0, "val": 'ok', "error": ""}))

# /video/videotask
def video_task(request):
    if request.method == 'POST':
        famous_list = request.POST.get('famous_list')
        videoid = request.POST.get('videoid')

        if not famous_list:
            return HttpResponse(json.dumps({"ret": 1, "val": 'Please input famousname', "error": ""}))

        if not videoid:
            return HttpResponse(json.dumps({"ret": 1, "val": 'Please input media_id', "error": ""}))

        media_query = models.Media.objects.filter(media_id=videoid).first()
        media_path = MEDIA_ROOT + media_query.file_path
        media_name = media_query.name

        task = models.Task(name=media_name, task_media_id=videoid, target_face_id=famous_list)
        task.save()
        task_id = task.task_id
        # mgr_task.feed((file_url, media_id))
        famous_list = json.loads(famous_list)
        face_db = {}
        for famous in famous_list:
            famous_id = int(famous)
            famous_query = models.Famous.objects.filter(id=famous_id).first()
            if not famous_query:
                return HttpResponse(json.dumps({"ret": 1, "val": 'invalid famous', "error": ""}))
            famous_name = famous_query.name

            face_query = models.Face.objects.filter(famous_id=famous_id, is_show=True).all()
            feat_list = []
            for face in face_query:
                face_feat = face.feat
                feat_list.append(decode_feat(face_feat))

            face_db[famous_id] = (famous_name, feat_list)

        #mgr_task.feed((media_path, face_db, task_id))
        data = {"media_path":media_path, "face_db":face_db, "task_id":task_id}
        jdata = {"data": json.dumps(data)}
        detection_task_add(jdata)
        return HttpResponse(json.dumps({"ret": 0, "val": '', "error": ""}))

# /video/upload
def videoupload(request):
    if request.method == "POST":
        type = request.POST.get('type')
        if type == 'url':
            external_id = request.POST.get('external_id')
            external_system = request.POST.get('external_system')
            videoname = request.POST.get('video_name')
            file_url = request.POST.get('file_url')
            time_str = time.time()
            media_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, str(time_str)))
            opener = urllib2.build_opener()
            try:
                opener.open(file_url)
                models.Media.objects.create(media_id=media_id, external_id=external_id, external_system=external_system, name=videoname, media_status='上传中', force_url=file_url)
                mgr.feed((file_url, media_id))
                return HttpResponse(json.dumps({"ret": 0, "val": "success", "error": ""}))
            except urllib2.HTTPError, e:
                #return HttpResponse(json.dumps({"ret": 1, "val": "HTTPError", "error": ""}))
                return HttpResponse(json.dumps({"ret": 1, "val": e.code, "error": ""}))
            except urllib2.URLError:
                return HttpResponse(json.dumps({"ret": 1, "val": "URLError", "error": ""}))
            except Exception as e:
                return HttpResponse(json.dumps({"ret": 1, "val": "UnknowError", "error": ""}))
        else:
            file = request.FILES.get('file')
            print(file)
            time_str = time.time()
            media_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, str(time_str)))

            models.Media.objects.create(media_id=media_id, media_status='本地上传中')
            mediaquery = models.Media.objects.get(media_id=media_id)
            upload(file, mediaquery)
            return HttpResponse(json.dumps({"ret": 0, "val": "success", "error": ""}))

# /video/insertfamous
def insert_famous(request):
    file_name = '/home/ysten/yzg/data/cpc/cpc_info.txt'
    with io.open(file_name, 'r', encoding='utf-8') as file_to_read:
        famous_list = []
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            # num, name, birthplace, birthday, height, university, works = [i for i in lines.split('    ')]
            data = [i for i in lines.split('    ')]
            datalist = [i for i in data[0].split('\t')]
            if len(datalist) < 3:
                print(datalist)
            if len(datalist) == 3:
                num = datalist[0]
                name = datalist[1]
                works = datalist[2]
                famous_list.append(
                    models.Famous(
                        num = num ,
                        name = name,
                        works = works,
                        type = 2
                    )
                )
                # models.Famous.objects.create(num=num, name=name, birthplace=birthplace, birthday=birthday, height=height, university=university, works=works)

            if not lines:
                break
                pass
        models.Famous.objects.bulk_create(famous_list)
    return HttpResponse(json.dumps({"ret": 0, "val": "success", "error": ""}))


# /video/getpicfeat
def get_pic_feat(request):
    feat_file = '/home/ysten/yzg/data/cpc/cpc_feats1.pkl'
    pkl_file = open(feat_file, 'rb')
    feat_dict = pickle.load(pkl_file)
    famous_query = models.Famous.objects.filter(type=2).all()
    face_list = []
    for famous in famous_query:
        famous_num = famous.num
        famous_id = famous.id
        famous_num_dict = feat_dict.get(famous_num)
        if not famous_num_dict:
            print(famous_num)
        else:
            for img_name, feat in famous_num_dict.items():
                url = "image/" + famous_num + '/' + img_name
                path = os.path.join(famous_num + '/'+ img_name)
                feat = feat
                famous_id = famous_id
                face_list.append(
                    models.Face(
                        url = url,
                        path = path,
                        feat = feat,
                        famous_id = famous_id
                    )
                )
    print(len(face_list))
    models.Face.objects.bulk_create(face_list)
    return HttpResponse(json.dumps({"ret": 0, "val": 'OK', "error": ""}))

# /video/changeurl
def change_url(request):
    face_query = models.Face.objects.filter()
    face_list = []
    for face in face_query:
        face_dict = {}
        face_url = str(face.url)
        face_path = str(face.path)
        face_id = face.id
        img_url = face_url.split(upload_url)[-1]
        img_path = face_path.split(IMAGE_ROOT)[-1]
        face_dict[face_id] = (img_url, img_path)
        face_list.append(face_dict)
    for face_dict in face_list:
        for face_id, changedata in face_dict.items():
            print(face_id)
            print(changedata)
            face_query = models.Face.objects.filter(id=face_id)
            face_query.update(url=changedata[0], path=changedata[1])
    return HttpResponse(json.dumps({"ret": 0, "val": "success", "error": ""}))
