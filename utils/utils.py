import numpy as np
from PIL import Image
WIDTH = 84
HEIGHT = 84


#def set_frame_size(height, width):
    #def inner(func):
        #return 


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def pil_gray_scale(frame):
    return np.array(Image.fromarray(frame).resize((WIDTH, HEIGHT)).convert('L'))
    #return Image.from_array(frame).resize(WIDTH, HEIGHT).convert('L')

def numpy_gray_scale(frame):
    # return mean(array(img[::2, ::2]), axis=2).astype(np.uint8) # TODO why does this leak memory?
    return np.dot(np.array(frame), [0.299, 0.587, 0.144])[::2, ::2].astype(np.uint8)


def down_numpy_gray_scale(frame):
    return np.dot(np.array(frame)[::2, ::2], [0.299, 0.587, 0.144]).astype(np.uint8)


def down_numpy_mean(frame): # TODO WHY THE F** DOES THIS LEAK MEMORY
    np_frame = np.array(frame)
    downsized_frame = np_frame[::2, ::2]
    gray_scaled_frame = np.mean(downsized_frame, axis=2)
    casted_frame = gray_scaled_frame.astype(np.uint8)
    #del frame
    return casted_frame
    #return np.mean(np.array(frame)[::2, ::2], axis=2).astype(np.uint8)


def numpy_mean(frame): # TODO WHY THE F*** DOES THIS LEAK MEMORY
    # return mean(array(img[::2, ::2]), axis=2).astype(np.uint8) # TODO why does this leak memory?
    # #########jreturn np.mean(np.array(frame), axis=2)[::2, ::2].astype(np.uint8)
    #return np.mean(np.array(frame), axis=2)[::2, ::2].astype(np.uint8)
    return np.mean(frame, axis=2)[::2, ::2].astype(np.uint8)


def convert_atari_frame(frame):
    
    ##return numpy_gray_scale(frame)
    #return down_numpy_mean(frame)
    #return numpy_mean(frame)
    #return np.mean(np.array(frame), axis=2)[::2, ::2].astype(np.uint8)
    #return down_numpy_gray_scale(frame)
    return pil_gray_scale(frame)
