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
    print(f"cpus: {os.cpu_count()}")
    # Envrionments and reward thesholds
    environments = [
        "CartPole-v0",
        #"CartPole-v1",
        #"MountainCar-v0",
        #"Acrobot-v1",
        #"LunarLander-v2",
    ]
    games_to_play = [
        400,
        #600,
        #1000,
        #500,
        #1000,
    ]
    learners = [
        DeepQFactory.create_vanilla_deep_q(),
        DeepQFactory.create_double_deep_q(),
        DeepQFactory.create_clipped_double_deep_q(),
        DeepQFactory.create_duel_deep_q(),
        DeepQFactory.create_double_duel_deep_q(),
        DeepQFactory.create_clipped_double_duel_deep_q()
        #SuperQ,
    ]
    # env = gym.make("LunarLander-v2")
    # env = gym.make("MountainCar-v0")
    env = gym.make("CartPole-v0")
    all_envs = gym.envs.registry.all()
    # deterministics = [env_spec for env_spec in all_envs if not env_spec.nondeterministic]
    ids = [env_spec for env_spec in all_envs if env_spec.id in environments]
    games = [(env_spec.id, env_spec.max_episode_steps, env_spec.reward_threshold) for env_spec in all_envs if
             env_spec.id in environments]
    # games = [env_spec.id, env_spec.rew for env_spec in all]
    # sort list to match order in environments
    games = [item for name in environments for item in games if item[0] == name]
    temp = env.reward_range
    data = []
    # for name, max_episode_steps, reward_threshold in tqdm(games):
    for (name, max_episode_steps, reward_threshold), max_episodes in zip(games, games_to_play):

        # bar = tqdm(range(3), bar_format="{postfix[0]} {postfix[1][value]:>8.2g}", postfix=["Learner", dict(value=1]))] desc="Learners")
        # learner_meter = tqdm(range(3), desc="Learners")
        # for idx in learner_meter:
        for learner in learners:
            # bar.set_postfix(learner.get_name())
            # print(f"\nPlaying {name} with {learner.get_name()}\n")

            # if name != "LunarLander-v2":
            # continue
            print(f"\nStarting: {name} {learner.get_name()}")
            env = gym.make(name)
            feature_count = env.observation_space.shape[0]
            action_count = env.action_space.n
            gamma = float(
                np.power(0.0001, 1. / max_episode_steps))  # Scale gamma to approach zero near max_episode_steps
            learner.build_model(input_dimension=feature_count, output_dimension=action_count,
                                nodes_per_layer=128,
                                learning_rate=0.001,
                                #learning_rate=0.0001,
                                layer_count=2,
                                gamma=gamma),
            start_length = int(max_episodes/10) * max_episode_steps
            # TODO account for possible extra space from scoring
            max_possible_step_count = (max_episodes * max_episode_steps) + start_length

            agent = Agent(
                learner=learner,
                scorer=Scores(10),
                sample_size=64,
                #sample_size=128,
                #replay_buffer=ReplayBuffer(max_length=1000 * max_episode_steps, start_length=start_length),
                replay_buffer=ReplayBuffer(max_length=max_possible_step_count, start_length=start_length),
                environment=env,
                reward_threshold=reward_threshold,
                random_choice_decay_min=0.00,
                max_episode_steps=max_episode_steps,
                max_episodes=max_episodes,
                early_stopping=False,
                verbose=0)
            step_count = agent.play(4000 * max_episode_steps, verbose=0)
            score = agent.score_model(100, verbose=0)
            # print(f"\n------------ FINAL Average reward: {score} -----------")

            # agent.plot(name, learner.get_name())
            data.append((learner.get_name(), name, step_count, score))
            # learner_meter.write(f"{learner.get_name()} Done. Final Average Score: {score}")
            del agent  # Currenlty required to cleanup tqdm status bars when verbose > 0
            print(f"{name} {learner.get_name()} Done. Final Average Score: {score}. Step_count = {step_count}\n")
            #exit(1)

    print(data)

    # agent.learner.model.load_weights("model.h5")
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
