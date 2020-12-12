import numpy as np
from PIL import Image
import tensorflow as tf
WIDTH = 84
HEIGHT = 84

#sess = tf.compat.v1.Session()

#def set_frame_size(height, width):
    #def inner(func):
        #return 


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def pil_crop_gray_scale(frame):

    #width, height = frame.size
    #left =
    #img = Image.fromarray(frame)
    #width, height = img.size
    #left = 0
    #right = width
    #top =
    #return np.array(Image.fromarray(frame).convert('L').crop((
        #0, 160+34, 160, 34)))
    return np.array(Image.fromarray(frame).convert('L').crop((0, 34, 160, 194)).resize((WIDTH, HEIGHT)))
    #return np.array(Image.fromarray(frame).convert('L').crop((0, 160+34, 160, 34)))
    #return np.array(Image.fromarray(frame).convert('L').crop((0, 160+34, 160, 34)))


def pil_gray_scale(frame):
    return np.array(Image.fromarray(frame).resize((WIDTH, HEIGHT)).convert('L'))
    #return np.array(Image.fromarray(frame).resize((WIDTH, HEIGHT)).convert('L'))
    #return np.array(Image.fromarray(frame).resize((WIDTH, HEIGHT)).convert('L'))
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

@tf.function
def tf_gray_scale(frame):
    gray_frame = tf.image.rgb_to_grayscale(frame)
    #return tf.image.resize(gray_frame, [HEIGHT, WIDTH])
    return tf.squeeze(tf.image.resize(gray_frame, [HEIGHT, WIDTH]))


#sess = tf.compat.v1.InteractiveSession()

@tf.function #TODO learn more about tf.function
def process_image(image):
    #image = tf.convert_to_tensor(image, dtype=tf.uint8)
    image_gray = tf.image.rgb_to_grayscale(image)
    #tf.print(tf.shape(image))
    #tf.print(tf.shape(image_gray))
    #print(tf.shape(image_gray))
    # https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
    image_cropped = tf.image.crop_to_bounding_box(image_gray,
                                                  offset_height=34,
                                                  offset_width=0,
                                                  target_height=160,
                                                  target_width=160)
                                                  #target_width=160)
    #image_rgb =  tf.cond(tf.rank(image) < 4,
                         #lambda: tf.image.grayscale_to_rgb(tf.expand_dims(image, -1)),
                         #lambda: tf.identity(image))
    # Add shape information
    #s = image.shape
    #image_rgb.set_shape(s)
    #if s.ndims is not None and s.ndims < 4:
        #image_rgb.set_shape(s.concatenate(3))
    temp = tf.image.resize(image_cropped, [HEIGHT, WIDTH],
                           method=tf.image.ResizeMethod.BILINEAR)
    temp = tf.squeeze(temp)
    # TODO use crop and resize method
    #return np.array(temp.eval()).squeeze(axis=-1)
    #return np.array(temp).squeeze(axis=-1)
    return temp
    #return image_rgb

#def test()
def test(image):
    return process_image(tf.constant(image))

def convert_atari_frame(frame):
    ##return numpy_gray_scale(frame)
    #return down_numpy_mean(frame)
    #return numpy_mean(frame)
    #return np.mean(np.array(frame), axis=2)[::2, ::2].astype(np.uint8)
    #return down_numpy_gray_scale(frame)
    #print(tf.shape(frame))
    #return np.array(process_image(frame))
    #return np.array(test(frame))

    #return process_image(frame).eval(session=sess)
    #return process_image(frame)
    #return tf_gray_scale(frame)


    return pil_crop_gray_scale(frame)
