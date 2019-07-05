# -*- coding: utf-8 -*-
import argparse
import os
import sys
import struct
import ctypes
import glob
import base64
import math
import json
import random
import numpy as np

DIST_THRESH = 0.25
UNKNOWN_ID  = -1

def cosine_similarity(v1, v2):
    # compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
    # print(v1)
    # print(v2)
    if len(v1) != len(v2):
        return -1
    else:
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]
            y = v2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y

        return sumxy / math.sqrt(sumxx * sumyy)


def cosine_distance(v1, v2):
    return (1 - cosine_similarity(v1, v2)) / 2


def load_feat_bin(path):
    feat = []
    with open(path, "rb") as f:
        chunk = f.read(4)
        n = struct.unpack('i', chunk)
        n = n[0]
        for i in range(n):
            chunk = f.read(4)
            d = struct.unpack('f', chunk)
            feat.append(d[0])
    # print('decode len:%d' % len(feat))
    return feat


def write_feat_bin(path, feat):
    with open(path, "wb") as f:
        chunk = struct.pack('i', len(feat))
        f.write(chunk)
        for i in range(len(feat)):
            chunk = struct.pack('f', feat[i])
            f.write(chunk)

def decode_feat(feat_str):
    try:
        feat_bytes = base64.b64decode(feat_str)
    except Exception as e:
        return []
    n = int(len(feat_bytes) / 4)
    # print('feat len:%d' % n)
    feat = []
    for i in range(n):
        d = struct.unpack('f', feat_bytes[i * 4:(i * 4 + 4)])
        feat.append(d[0])
    # print('decode len:%d' % len(feat))
    return feat


def compare_feat(a_str, b_str):
    fa = decode_feat(a_str)
    fb = decode_feat(b_str)
    return cosine_distance(fa, fb)


def compare_feat_array(a_str, b_list, drop_ratio=0.3):
    fa = decode_feat(a_str)
    dists = []
    for idx, b_str in enumerate(b_list):
        fb = decode_feat(b_str)
        d = cosine_distance(fa, fb)
        dists.append(d)
    end = int(len(b_list) * (1 - drop_ratio))
    end = max(1, end)
    # sort
    dists.sort()
    kept = dists[:end]
    avg = sum(kept) / end
    return avg

    
def decode_user_list(user_list):
    '''
    { id: feat_list }
    '''
    feat_dict = {}
    for id, l in user_list.items():
        feat_list = []
        for jstr in l:
            feat = decode_feat(jstr)
            feat_list.append(feat)
        feat_dict[id] = feat_list
    return feat_dict
    

def compare_with_user(fa, b_list, drop_ratio=0.3):
    dists = []
    for idx, fb in enumerate(b_list):
        d = cosine_distance(fa, fb)
        dists.append(d)
    end = int(len(b_list) * (1 - drop_ratio))
    end = max(1, end)
    # sort
    dists.sort()
    kept = dists[:end]
    avg = sum(kept) / end
    return avg

def compare_with_celeb(fa, b_list, topN=5):
    dists = []
    for idx, fb in enumerate(b_list):
        d = cosine_distance(fa, fb)
        dists.append(d)
    end = max(len(dists), topN)
    # sort
    dists.sort()
    kept = dists[:end]
    avg = sum(kept) / end
    return avg
    
    
def recognize_face_id(jdata, user_list):
    face_list = jdata.get('faceList')
    if not face_list or len(face_list) < 1:
        return jdata

    for face in face_list:
        feat = face.get('feat')
        feat = decode_feat(feat)
        # match with all
        min_dist = 100
        min_id   = -1
        for id, l in user_list.items():
            dist = compare_with_user(feat, l)
            if dist < min_dist:
                min_dist = dist
                min_id = id
        if min_dist < DIST_THRESH:
            face['id'] = min_id
            face['dist'] = min_dist
        else:
            face['id'] = UNKNOWN_ID
            face['dist'] = min_dist
        # remove feat field
        if 'feat' in face:
            face.pop('feat')
    return jdata

    
def cluster_feat_array(feat_strs, sim_thresh = 0.8):
    # extract feature
    feats = [np.array(decode_feat(x)) for x in feat_strs]
    n = len(feats)
    # clustering 'center': feat, 'id': ,'bestIdx': bestmatch index, 'members':[]
    clusters = []
    # user sim to cosine_similarity
    sim_thresh = 2 * (sim_thresh - 0.5)
    todo_set = set([i for i in range(n)])
    while(todo_set):
        # random select a seed
        #seed = random.choice(list(todo_set))
        seed = list(todo_set)[0]
        y = feats[seed]
        # find its neigbours
        c_set = set([seed])
        c_sum = y
        for idx in todo_set:
            if idx == seed:
                continue
            x = feats[idx]
            sim = cosine_similarity(y, x)
            if sim >= sim_thresh:
                c_set.add(idx)
                c_sum += x
        # refinement
        c_cnt = len(c_set)
        if c_cnt > 1:
            # refine with a second run
            y = c_sum/c_cnt
            # find its neigbours
            c_set = set()
            max_sim = -1
            max_idx = -1
            sim_sum = 0
            sim_list = []
            for idx in todo_set:
                x = feats[idx]
                sim = cosine_similarity(y, x)
                if sim >= sim_thresh:
                    c_set.add(idx)
                    sim_list.append(sim)
                    sim_sum += sim
                    if sim > max_sim:
                        max_sim = sim
                        max_idx = idx
            bestIdx = max_idx
            avg_sim = sim_sum/len(c_set)
        else:
            bestIdx = seed
            avg_sim = 1.0
            sim_list = [1.0]
        # add the new cluster
        cluster = {}
        cluster['id'] = len(clusters)
        # cluster['center'] = y
        cluster['members'] = list(c_set)
        cluster['bestIdx'] = bestIdx
        # cluster['avgSim'] = avg_sim
        # cluster['sims'] = sim_list
        clusters.append(cluster)
        # remove from todo
        todo_set = todo_set - c_set
    return clusters


def compare_feat_arreries(params):
    a_list = params
    n = len(a_list)
    p2d = np.ones([n, n])
    feats = [decode_feat(x) for x in a_list]
    for i in range(n):
        for j in range(n):
            resu = cosine_distance(feats[i], feats[j])
            p2d[i,j] = '%.2f' % ((1-resu)*100)
    p2d_list = p2d.tolist()
    return p2d_list

if __name__ == '__main__':
    feat = ["SYYtwJod3b4KsWs/8Pqsv3UDnD4AksS/qMZmv9F7vj++Bxi+iudOvZZ0uD+8ORg+elWQPypRLz7fg5M/99oEPlc/Dz87w7C+WP7Mv1m/XD+0SXU+DTmuvszKk78JPb6/RMCEv+tNlL5v59e/modjP9U6gr8grQ+/6K2DPyYqx76AEtq/viKJPldlmL+QVF4/CuvCv38vvr8d4xY/UGuRvzdVij9dg7k/+uQyPxiLo794y8Y/0EciQKTMVTwrNTS/jzpMP2Clm7/bfFm/6znlv4Pqyb12J54/fltEv9x4sr+CM0C/UGQqP69vn7+TNVU+82+sv0gfgL+Pd+o+66WEv4N6iL/Px1e/Kd7YP1sDiL+7CdU/UGPrvw6EED9TEB4/fAHxv/kgob9G2XM/nte6PgH9ML9PIKK+sMTBP+vkYT9Ee3c/xh7SP64Jzr7OQpO+kwQiwCSG5z+ACCs/CD3uu/T0wT9uXnK/KmJ6PpwBMcD/OJ6/EXOfvzRNnb7srWS8x7tCP5P5n71oweC/Eq1CPz3TIL9BHnm+N+kMwJXerj9o8aE/QZS2P0dVVz9EWyk+UnwIwEEv9r4sAYU+ZWCJP2UF3T4oF+w/WZ4tvmJjt7wb7+G/aQ5KPyKIkz7OTPQ+y6G8vZmy874ifhhAMKwcvzlY1z7Flzo/0af+PhB39z4=",
            "e0x0vzAAGz8cqZM/RAYTPkwGpj9oTHW+OIjBv+PdmL/KQUY+ymcvv9sZdb1AorK/YlQyP7t4KT9e1Zc/bctvP23S2z/C3BrA2I52v+QUvr8+0zC/jfzVP2HOkz7Udxm/NkN5v07/XL4o3Fm/ULURvngrzT7ZJxa/kiBCv6ucHMADgTVAl/sIv4FIWb4O/FS/tWMgv5x09z1cRCW+1E3CvwjkPr/3KNS/U5vTPph5tb4/10s+NaOdv8sTT74h/Cc+qQjAPh21DcBiE6Y/mkzuvz4+q7wBR80/cXbSPH2vYL98yQs/TBP2Pnsm6L4idQG/EVRDPsqxx7y7n5C+r1mzvquSh72xoxk/tCWVv9P5pTyezcW/4bMsP33klT/XQVs/sDskQOTeOj3vgxE/zrHjvuCqlD+Bd34/h8g0v8wwGcDhQ9W/VE4iPmVXbj/Ln16+ShU8PtPBPz+yixs/E1rpPj9bE78+bvU/2HQGvqSeg7/XnfG+cPWmv1aAF78y+4o9snmPP1trH74YnyRANwLDPh/xeb/ngYI+lumsPxrtgT8+o5e9PArsvs92R79mswu/YY2Iv9pWPr/hVXq/BFwrPolChj1JVJW/TxFovp1GKj9foJc+WFVbvw5Sa75Xe/K+Hg/ePel+Hj+HTAtAyGv8PkK+JT1TZI6/t7REv+L9Rb8=",
            "g4r9vi4gSj7jIHQ/EvOOP49nRL9ixBG+LO+Vvi1NcL7VWpO/e6fUvmN2GL+z7ya/j2m9P+2Zlj/NBm6/SONKP7VE3L+oQZe/l6cLPjzglj/Q3s+/r7mlPnq+5T4YD4Q/0Oz8PqF5V72DF/E9X/uLv8x1Nz45eWw+huEbv4qFaz97PkG/sJfZPQPDeD/OO6W/badRvsXYiz/GbpE/uDW+PlOk0zwWOlW+eX/nPlu0g78qT9i9InWxPn66ij92s4e+HPnVvvnOBD8bDeQ+A7tbv3/DBDwJvgC+eTySv9gG977Sj92/tyt2P6aOoj/GoBq/hptlvwYfOL+solm/Q6+RvV4nBz/bP4G/4rqjvn32tz4V8J0+HNJFP1dLTT3lTwy/8UKVv4F76r5W1ZW/3oFDPyJ9N78OEk+/COYAPmFwlj/Be2W/wx4cPgAIgT8aj649ImOWPge6WD9dp+C9tmGWPhLI0D7mCWC/t0T+O+aujr5g18y/Y/OSP36QLz+wny+89BKOvUBrFL/jZjW/uYtVv0dzJb4/oTE/KV7JvUBPm761rgA/Ea/DPyt71b+Wb5s+4aKoP6B+br6KBYg/1xZlP2gkUD+Jq9U/EnGnPwzHqj5iRTK+DyPYPkYKwb971gZAVN+Dv7BdiL/8KJE+ME4CP28aa78YCQw/HvJZP6JD5r4="]
            
    fa = "rXcDv2kL2r6ilFQ+9nS4vzdwBkCyvCW/WG+dP3hSTT63lX++VHTHvx87JL444LG+IBe5P/tij7/kdHm8cNUUv1vU5r+iBSq/Gi4SPI0wJj6e/ja/mrCwv/Pdvz/vc+W+QZCWP/SlYj/FUg7AzwPnPigG9r67dUu+pdcOQMAeA78nNGm/yHPyvp/KbL9nXlG9jYgNwBkdQj8KU8C+lhIKvrSTJD3LUtu/PttDP1aSQT+8QIc/1Nihv2uUOT9kjeY/K+TTvpafZj8xvFe+HwmuPhzMWL92994+Cw6uP/RAxjoHJK8/2gK0PjO2mj5cSLO/k8ODvpJoFr+vML0/RD9rPzxUX79/f8M/3XnFPx+8yD5or+m9MkPkPXUwnD/ky5g/zi7nPiqhHz/+RYK/KtiJv2CrrL+W2H2/tHKTPiY9Lr6k7eM9PgPWP1mUKL5Eh8w/WqO8vxHaxz5oR9W91haOPvZPfb8o3pu9ITfaP+cA4b5+jAO/aXAKP+peGL4tVZY/YG6uvlov4794gdM+m/SUvg8cmT9Z7pu/DbtkPxMV9r4LfAU+0puRPxKpRD7RmcE/Y/Ybv2/fzb5lMQtAXN80P+Lzkj8g7Ms+BUiRPzqbHj/SpYA/rMa0PtgFYL87sdc+smiMP5Bws74MsKg+gZ4Dv56GBkBadx2/71S2P7Aakr8="
    fb = "/PYRv+OFhD8K+1y+MsK+vwLbVj8cnCk/V2C+PhaAwT+dueW+4uXSv1t01b8tWLO9hD0LPbfrR79MiRu8QsjJv+wmcr9Ym42/4yhgv5Xlkj/dUuE+MDeIv+113j8y06w+2ZArv5pxXD6SbwO/lnzBP+56Jz7Hxjk+qezkPw0GpL84H829gfiCv8YKs79j11K+iyw1vpzmzj6NXfS+Bj3OvxsTcbzcA5C/Auv1P3tUer3wnOs/6YqXP0vSd7/336k/yUyOvQMPej+T5jvAL6XMPogwZT8zVE8/k6+sP79fGL7RIfs/AIVXOnNDW765Pve/W/NCv4P0I77b4mM/Lq3GPvhnlD6OlkQ/7mB0P8vQjD/beAY//DJWPiDHQz4O4qU9TTZvPtXrhT/YwOa/aShbvjrpBMBFDsO+PWMEv25Scr7cYJE9qzLXPhTzMr/jmi8+8Ci8v8JGV79kDJG+A4MXv4PyPj+WTAW/+GbtPM6kNb9wSaS/Sn7RP9tsRz8k5iW/bIaqPpXhkL/BHhW+580lP22eUj8glXO/LhWLPy2eqL8ZF1A/aDKRPp+d174c+RdAwVp/vq9EHr/5COw/HWfRP426I7/3YIg/SeSeP+maCT9Knrw/TitIP1wo0r/kI+E7RQXBvXmZ876Flno/gNUqP1wPxbwj8L2/nw1VPwkSlL4="
    print(1-compare_feat(fa, fb))
    resu = cluster_feat_array(feat)
    print(resu)