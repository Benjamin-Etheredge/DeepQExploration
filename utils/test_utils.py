from unittest import TestCase

from utils import convert_atari_frame
import gym
from PIL import Image

class Test(TestCase):
    def view_convert_atari_frame(self):
        env = gym.make("SpaceInvaders-v4")
        frame = env.reset()
        image = Image.fromarray(frame)
        image.show()
        converted = Image.fromarray(convert_atari_frame(frame))
        converted.show()


