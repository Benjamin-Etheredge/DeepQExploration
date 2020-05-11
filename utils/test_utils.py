from unittest import TestCase

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from utils import convert_atari_frame
import gym
from PIL import Image

class Test(TestCase):
    def test_view_convert_atari_frame(self):
        env = gym.make("SpaceInvaders-v4")
        frame = env.reset()
        image = Image.fromarray(frame)
        tmp = convert_atari_frame(frame)
        converted = Image.fromarray(tmp)
        image.show()
        converted.show()


if __name__ == "__main__":
    Test.view_convert_atari_frame()
