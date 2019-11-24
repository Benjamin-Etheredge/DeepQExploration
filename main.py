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

    # Envrionments and reward thesholds
    environments = [
        "CartPole-v1",
        "CartPole-v0",
        "MountainCar-v0",
        "Acrobot-v1",
        "LunarLander-v2",
        #"Acrobot-v1",
        ]
    #env = gym.make("LunarLander-v2")
    #env = gym.make("MountainCar-v0")
    env = gym.make("CartPole-v0")
    all_envs = gym.envs.registry.all()
    #deterministics = [env_spec for env_spec in all_envs if not env_spec.nondeterministic]
    ids = [env_spec for env_spec in all_envs if env_spec.id in environments]
    games = [(env_spec.id, env_spec.max_episode_steps, env_spec.reward_threshold) for env_spec in all_envs if env_spec.id in environments]
    #games = [env_spec.id, env_spec.rew for env_spec in all]
    # sort list to match order in environments
    games = [item for name in environments for item in games if item[0] == name]
    temp = env.reward_range
    data = []
    for name, max_episode_steps, reward_threshold in games:
        for idx in range(3):
            learner_idx = idx % 3
            if learner_idx == 0:
                learner = DeepQ
            elif learner_idx == 1:
                learner = DoubleDeepQ
            elif learner_idx == 2:
                learner = DuelDeepQ
            print(f"\nPlaying {name} with {learner.get_name()}\n")

            #if name != "LunarLander-v2":
                #continue
            env = gym.make(name)
            feature_count = env.observation_space.shape[0]
            action_count = env.action_space.n
            gamma = float(np.power(0.0001, 1./max_episode_steps))
            agent = Agent(
                learner=learner(input_dimension=feature_count,
                                  output_dimension=action_count,
                                  nodesPerLayer=128,
                                  numLayers=2,
                                  gamma=gamma),
                scorer=Scores(100),
                replayBuffer=ReplayBuffer(max_length=1000*max_episode_steps, start_length=300*max_episode_steps),
                environment=env,
                reward_threshold=reward_threshold,
                random_choice_decay_min=0.01,
                max_episode_steps=max_episode_steps,
                verbose=0)
            iteration_count = agent.play(4000 * max_episode_steps, verbose=1)
            score = agent.score_model(150, verbose=0)
            print(f"\n------------ FINAL Average reward: {score} -----------")

            agent.plot(name, learner.get_name())
            data.append((learner.get_name(), name, iteration_count, score))

    print(data)

    #agent.learner.model.load_weights("model.h5")
    # pr.enable()
    '''
    start_time = timer()
    agent.score_model(400)
    end_time = timer()
    print(f"took: {round(end_time - start_time, 2)}s")
    start_time = timer()
    agent.score_model(400)
    end_time = timer()
    print(f"took: {round(end_time - start_time, 2)}s")
    start_time = timer()
    agent.score_model(400)
    end_time = timer()
    print(f"took: {round(end_time - start_time, 2)}s")
    '''

    # pr.disable()
    agent.learner.model.save_weights("model.h5")


    # pr.dump_stats('profil.pstat')
