# from django.urls import path
from django.conf.urls import url
from . import views

urlpatterns = [
    url('home', views.index, name='index'),
    url('videoadmin', views.video_admin, name='admin'),
    url('targetrec', views.target_recognition, name='recognition'),
    url('famoustask', views.famoustask, name='famoustask'),
    url('politicstask', views.politicstask, name='politicstask'),
    url('targetmag', views.target_management, name='management'),
    url('showvideo', views.showvideo, name='showvideo'),
    url('searchvideo', views.search_video, name='searchvideo'),
    url('upload', views.videoupload, name='upload'),
    url('getfacelist', views.get_face_list, name='getFace'),
    url('addface', views.add_face, name='addFace'),
    url('autoaddface', views.auto_add_face, name='autoaddface'),
    url('getfacefeat', views.get_face_feat, name='getfacefeat'),
    url('newfamous', views.new_famous, name='newfamous'),
    url('delface', views.del_face, name='delface'),
    url('delvideo', views.del_video, name='delvideo'),
    url('changevideo', views.change_video, name='changevideo'),
    url('getvideo', views.get_video, name='getvideo'),
    url('getfamous', views.get_famous, name='getfamous'),
    url('changefamous', views.change_famous, name='changefamous'),
    url('delfamous', views.del_famous, name='delfamous'),
    url('videotask', views.video_task, name='videotask'),
    url('keyframetask', views.key_frame_task, name='keyframetask'),
    #url('insertfamous', views.insert_famous, name='insertfamous'),
    url('deltask', views.del_task, name='deltask'),
    url('getfacepic', views.get_facepic, name='getfacepic'),
    url('getkeyframeprogress', views.key_frame_progress, name='getkeyframeprogress'),
    url('getkeyframestate', views.key_frame_state, name='getkeyframestate'),
    url('getdetectiontaskprogress', views.detection_task_progress, name='getdetectiontaskprogress'),
    url('getdetectiontaskstate', views.detection_task_state, name='getdetectiontaskstate'),


    #url('getpicfeat', views.get_pic_feat, name='getpicfeat'),
    #url('changeurl', views.change_url, name='changeurl'),
]

