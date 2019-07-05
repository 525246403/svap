#encoding:utf-8
from django.db import models
from django.utils import timezone

import datetime
# Create your models here.

class Media(models.Model):
    STATE_CHOICES = (
        (1, '准备检测'),
        (2, '检测中'),
        (3, '完成'),
        (4, ''),
    )
    media_id = models.CharField(max_length=64, primary_key=True, verbose_name="视频ID")
    external_id = models.CharField(max_length=64, verbose_name="外部ID", default="")
    external_system = models.CharField(max_length=64, verbose_name="外部系统", default="")
    name = models.CharField(max_length=30, verbose_name='名称', default="")
    file_url = models.CharField(max_length=128, verbose_name='访问地址')
    force_url = models.CharField(max_length=128, verbose_name='视频源地址',default="")
    video_format = models.CharField(max_length=16, verbose_name='视频格式', default="")
    size = models.CharField(max_length=16, verbose_name='视频大小', default="")
    video_length = models.CharField(max_length=16, verbose_name='视频长度', default="")
    file_path = models.CharField(max_length=128, verbose_name='存放地址', default="")
    media_status = models.CharField(max_length=32, verbose_name='状态', default="")
    create_time = models.DateTimeField(verbose_name='更新时间', auto_now=True)
    is_show = models.BooleanField(default=True, verbose_name='是否显示')
    progress = models.CharField(max_length=10, verbose_name='进度', default="")
    is_keyframe = models.BooleanField(default=False, verbose_name='是否有关键帧')
    keyframe_progress = models.CharField(max_length=10, verbose_name='关键帧任务进度', default="")
    keyframe_status = models.SmallIntegerField(choices=STATE_CHOICES, default=4, verbose_name='关键帧任务状态')

    class Meta:
        db_table = "tb_media"
        verbose_name = '视频信息    '
        verbose_name_plural = verbose_name

class Famous(models.Model):
    TYPE_CHOICES = (
        (1, '明星'),
        (2, '政治人物'),
        (3, ''),
    )
    name = models.CharField(max_length=30, verbose_name='名字', default="")
    num = models.CharField(max_length=30, verbose_name='数字', default="")
    birthplace = models.CharField(max_length=40, verbose_name='出生地', default="")
    birthday = models.CharField(max_length=30, verbose_name='生日', default="")
    height = models.CharField(max_length=8, verbose_name='身高', default="")
    university = models.CharField(max_length=60, verbose_name='大学', default="")
    works = models.TextField(verbose_name='作品', default="")
    updata_time = models.DateTimeField(verbose_name='更新时间', auto_now = True)
    is_show = models.BooleanField(default=True, verbose_name='是否显示')
    type = models.SmallIntegerField(choices=TYPE_CHOICES, default=3, verbose_name='人物分类')

    class Meta:
        db_table = "tb_famous"
        verbose_name = '名人信息    '
        verbose_name_plural = verbose_name

class Face(models.Model):
    updata_time = models.DateTimeField(verbose_name='更新时间', auto_now = True)
    feat = models.TextField(verbose_name='特征值', default="")
    url = models.CharField(max_length=128, verbose_name='链接', default="")
    path = models.CharField(max_length=128, verbose_name='地址', default="")
    famous = models.ForeignKey('Famous', on_delete=models.CASCADE)
    is_show = models.BooleanField(default=True, verbose_name='是否显示')
    class Meta:
        db_table = "tb_face"
        verbose_name = '人脸信息    '
        verbose_name_plural = verbose_name

class Task(models.Model):
    STATE_CHOICES = (
        (1, '完成'),
        (2, '未开始'),
        (3, '检测中'),
    )
    task_id = models.AutoField(primary_key=True, verbose_name="任务ID")
    task_media = models.ForeignKey('Media', on_delete=models.CASCADE)
    name = models.CharField(max_length=30, verbose_name='名字', default="")
    state = models.SmallIntegerField(choices=STATE_CHOICES, default=2, verbose_name='状态')
    target_face_id = models.TextField(verbose_name='目标人物', default="")
    progress = models.CharField(max_length=10, verbose_name='进度', default="")
    create_time = models.DateTimeField(verbose_name='创建时间', auto_now_add = True)
    famous_list = models.ManyToManyField("Famous")
    is_show = models.BooleanField(default=True, verbose_name='是否显示')
    class Meta:
        db_table = "tb_task"
        verbose_name = '任务信息    '
        verbose_name_plural = verbose_name


class Count(models.Model):
    create_time = models.DateField(verbose_name='创建时间', auto_now = True)
    famous_count = models.IntegerField(verbose_name='明星总数', default="")
    face_count = models.IntegerField(verbose_name='明星图片总数', default="")
    media_count = models.IntegerField(verbose_name='视频总数', default="")
    task_count = models.IntegerField(verbose_name='任务总数', default="")
    class Meta:
        db_table = "tb_count"
        verbose_name = '计数信息'
        verbose_name_plural = verbose_name