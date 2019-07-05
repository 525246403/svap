
import os
import math
import time
import timeit
import numpy as np
import cv2


__all__ = ['clock', 'mad', 'psnr', 'ratio_sim', 'sobel_grad', 'tenen_score',
           'color_score', 'zip_score', 'best_score', 'hist_score',
           'list_dir', 'list_files']


def clock(func, debug=False):
    def clocked(*args, **kwargs):
        t0 = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed = timeit.default_timer() - t0
        if debug:
            name = func.__name__
            print('[%0.3fms] %s' % (elapsed*1000, name))
        return result

    return clocked


def brisque(frame, model_path, range_path):
    # brisque = cv2.quality.QualityBRISQUE_create(model_path, range_path)
    # score = brisque.compute([frame])

    score = cv2.quality.QualityBRISQUE_compute([frame], model_path, range_path)
    return score[0]


def salient_ft(src):
    """
    Frequency-tuned salient detection
    :param Src:
    :return:
    """
    src = cv2.GaussianBlur(src, (3, 3), sigmaX=0, sigmaY=0)

    Lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    Lab = Lab.astype(np.float32)

    Lab_t = np.transpose(Lab, [2, 0, 1])

    MeanL = np.mean(Lab_t[0])
    MeanA = np.mean(Lab_t[1])
    MeanB = np.mean(Lab_t[2])

    means = np.array((MeanL, MeanA, MeanB))
    means = means.reshape(1, 3)
    means = means.astype(np.float64)
    Lab = cv2.subtract(Lab, means)
    Lab = cv2.pow(Lab, 2)

    DistMap = np.sum(Lab, 2)
    DistMap = np.sqrt(DistMap)

    SaliencyMap = np.zeros(DistMap.shape, DistMap.dtype)
    cv2.normalize(DistMap, SaliencyMap, norm_type=cv2.NORM_MINMAX)

    SaliencyMap *= 255
    SaliencyMap = SaliencyMap.astype(np.uint8)

    return SaliencyMap


def reblur(src):
    if src.ndim == 3 and src.shape[2] == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    dst = cv2.boxFilter(src, -1, (5, 5), anchor=(-1, -1), normalize=True)

    # dst	= cv2.GaussianBlur(src, (5, 5), sigmaX=8, sigmaY=8)

    # cv2.imwrite('../test/src.png', src)
    # cv2.imwrite('../test/dst.png', dst)

    # src = cv2.medianBlur(src, 5)
    # dst = cv2.medianBlur(dst, 5)

    src_edge = sobel_grad(src)
    dst_edge = sobel_grad(dst)

    # src_edge = cv2.Sobel(src, cv2.CV_32F, 1, 1)
    # dst_edge = cv2.Sobel(dst, cv2.CV_32F, 1, 1)
    #
    # src_edge = np.absolute(src_edge)
    # dst_edge = np.absolute(dst_edge)
    # src_edge = np.uint8(src_edge)
    # dst_edge = np.uint8(dst_edge)

    # cv2.imwrite('../test/src_edge.png', src_edge)
    # cv2.imwrite('../test/dst_edge.png', dst_edge)
    # # cv2.imwrite('../test/src_edge_gray.png', src_edge)
    # # cv2.imwrite('../test/dst_edge_gray.png', dst_edge)

    # src_edge = cv2.cvtColor(src_edge, cv2.COLOR_BGR2GRAY)
    # dst_edge = cv2.cvtColor(dst_edge, cv2.COLOR_BGR2GRAY)

    # sim = cv2.matchTemplate(src_edge, dst_edge, cv2.TM_CCOEFF_NORMED)
    # score = 1.0 - sim[0, 0]

    # s = psnr(255., 255. - mad(src_edge, dst_edge))
    s = psnr(255., mad(src_edge, dst_edge))
    score = s
    #
    # sim, _ = cv2.quality.QualityGMSD_compute([src_edge], [dst_edge])
    # sim, _ = cv2.quality.QualityGMSD_compute([src], [dst])
    # score = sim[0]
    # sim, _ = cv2.quality.QualitySSIM_compute(src_edge, src_edge)
    # score = 1.0 - sim[0]

    # return score, src, dst
    return score, src_edge, dst_edge


def sobel_grad(image):
    image = image.astype(np.float32)
    sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    # sobelx = np.fabs(sobelx)
    # sobely = np.fabs(sobely)
    # sobelxy = (sobelx >> 1) + (sobely >> 1)
    sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    # sobelxy11 = cv2.Sobel(image, cv2.CV_32F, 1, 1)
    # sobelxy11 = cv2.convertScaleAbs(sobelxy11)

    # cv2.imshow('a', image)
    # cv2.imshow('sobelxy', sobelxy)
    # cv2.imshow('sobelxy11', sobelxy11)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # return sobelxy11
    return sobelxy


def mad(ref, cmp):
    diff = cv2.subtract(ref, cmp, dtype=cv2.CV_32F)
    diff = cv2.convertScaleAbs(diff)
    m = np.mean(diff)
    return m


def entropy(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0, 1], mask=None, histSize=[30, 32], ranges=[0, 180, 0, 256])
    size = hsv.shape[0] * hsv.shape[1]
    hist /= size
    hist = hist.flatten()

    hist += 1e-10
    entr_list = [p*math.log(p) for p in hist]
    # entr_list = [p*math.log(p) for p in hist if p > 0]
    entr = -sum(entr_list)

    return entr


def psnr(s, n):
    """
    :param s:signal(e.g. max value of image possible)
    :param n: noise(e.g. error of image)
    :return:
    """
    if n == 0:
        p = 1e10
    else:
        p = 20 * math.log(s / n, 10)
    return p


def ratio_sim(m, s):
    """
    :param s:signal
    :param m: max value of signal
    :return: similarity of range [0., 1.]
    """
    sim = s / m
    return sim


@clock
def tenen_score(image):
    sobelxy = sobel_grad(image)
    grad = np.mean(sobelxy)
    score = ratio_sim(128., grad)
    # score = psnr(256., 256. - grad)
    return score


def color_score(image):
    s = np.std(image, axis=2)
    m = np.mean(s)
    # score = psnr(120., m)
    score = psnr(121., 121. - m)
    return score


@clock
def bright_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright = np.mean(gray)
    score = ratio_sim(255., bright)
    return score


@clock
def contrast_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size = gray.size
    low = size * 0.1
    high = size * 0.9
    hist = cv2.calcHist([gray], [0], mask=None, histSize=[256], ranges=[0, 256])
    hist_cum = np.cumsum(hist)
    low_loc = 0
    high_loc = 255
    for i in range(size):
        if hist_cum[i] >= low:
            low_loc = i
            break
    for i in range(size):
        if hist_cum[i] >= high:
            high_loc = i
            break
    contrast = high_loc - low_loc
    score = ratio_sim(255., contrast)
    return score


@clock
def zip_score(image):
    status, buff = cv2.imencode('.png', image)
    s = len(buff)
    l = image.size
    score = ratio_sim(l, s)
    # score = psnr(l, l - s)
    return score


@clock
def hist_score(image):
    e = entropy(image)
    score = ratio_sim(6.87, e)
    # score = psnr(6.87, 6.87 - e)
    return score


@clock
def best_score(image):
    score_list = [tenen_score(image), hist_score(image), zip_score(image), bright_score(image), contrast_score(image)]
    score_mean = [0.26737832563685243,
                  0.6625887796176737,
                  0.4855895565106025,
                  0.24958390048994447,
                  0.5367090749261726]
    score_std = [0.10202863100240268,
                 0.0871418299704545,
                 0.10066312747833495,
                 0.10336236666332729,
                 0.17769825443640666]
    score_list = [(score_list[i] - score_mean[i]) / score_std[i] for i in range(5)]

    score = np.mean(score_list)
    return score, score_list


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )
    # files.sort()
    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files
