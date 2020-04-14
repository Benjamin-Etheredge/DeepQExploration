import numpy as np
#from PIL import Image


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def numpy_gray_scale(frame):
    # return mean(array(img[::2, ::2]), axis=2).astype(np.uint8) # TODO why does this leak memory?
    return np.dot(np.array(frame), [0.299, 0.587, 0.144])[::2, ::2].astype(np.uint8)


def down_numpy_gray_scale(frame):
    return np.dot(np.array(frame)[::2, ::2], [0.299, 0.587, 0.144]).astype(np.uint8)


def down_numpy_mean(frame):
    return np.mean(np.array(frame)[::2, ::2], axis=2).astype(np.uint8)


def numpy_mean(frame):
    return np.mean(np.array(frame), axis=2)[::2, ::2].astype(np.uint8)


def convert_atari_frame(frame):
    #return numpy_gray_scale(frame)
    return down_numpy_mean(frame)
