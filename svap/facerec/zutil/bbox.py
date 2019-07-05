#!/usr/bin/python
# -*- coding: UTF-8 -*-
import math


def IoU(A, B):
    '''
    Function:
        calculate Intersect of Union
    Input:
        rect_1: 1st rectangle
        rect_2: 2nd rectangle
    Output:
        IoU
    '''
    box_A = [float(A[0]), float(A[1]), float(A[2]), float(A[3])]
    box_B = [float(B[0]), float(B[1]), float(B[2]), float(B[3])]
    W = min(box_A[0]+box_A[2], box_B[0]+box_B[2]) - max(box_A[0], box_B[0])
    H = min(box_A[1]+box_A[3], box_B[1]+box_B[3]) - max(box_A[1], box_B[1])
    if W <= 0 or H <= 0:
        return 0
    SA = box_A[2] * box_A[3]
    SB = box_B[2] * box_B[3]
    intersection = W * H
    return intersection/(SA + SB - intersection)


def IoM(box_A, box_B):
    '''
    Function:
        calculate Intersect of Min area
    Input:
        rect_1: 1st rectangle
        rect_2: 2nd rectangle
    Output:
        IoM
    '''
    W = min(box_A[0] + box_A[2], box_B[0] + box_B[2]) - max(box_A[0], box_B[0])
    H = min(box_A[1] + box_A[3], box_B[1] + box_B[3]) - max(box_A[1], box_B[1])
    if W <= 0 or H <= 0:
        return 0
    SA = box_A[2] * box_A[3]
    SB = box_B[2] * box_B[3]
    min_area = min(SA, SB)
    intersection = W * H
    return intersection / min_area


def match_bbox(dts, gts, iou_thresh=0.5):
    # (dt, gt, iou)
    matched_dt = []
    fp_dt = []
    for i, dt in enumerate(dts):
        best_iou = 0
        best_gt = -1
        for j, gt in enumerate(gts):
            iou = IoU(dt, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = j
        if best_iou >= iou_thresh:
            matched_dt.append([i, best_gt, best_iou])
        else:
            fp_dt.append([i, -1, 0])
    return matched_dt, fp_dt
