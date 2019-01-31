#!/usr/bin/python3

# Custom Packages
from agent import *
from buffer import *
from learner import *
import logging

# create logger with 'spam_application'

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.warning('This will get logged to a file')


if __name__ == "__main__":
    import cProfile

    # pr = cProfile.Profile()
    # pr.disable()

    # agent = Agent(DeepQ, ReplayBuffer, 'LunarLander-v2')
    # agent = Agent(DoubleDeepQ, ReplayBuffer, 'LunarLander-v2')
    agent = Agent(DuelDeepQ, ReplayBuffer, 'LunarLander-v2')
    # pr.enable()
    agent.play()
    # pr.disable()

    # pr.dump_stats('profil.pstat')
