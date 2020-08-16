#!/usr/bin/python3

# Custom Packages
from agent import *
from buffer import *
import argparse
from utils.utils import convert_atari_frame

# create logger with 'spam_application'

#logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
#logging.warning('This will get logged to a file')
from learners import DeepQFactory

def get_env_info(env_name):
    all_envs = gym.envs.registry.all()
    ids = [env_spec for env_spec in all_envs if env_spec.id == env_name]
    env_spec = ids[0]
    return env_spec.id, env_spec.max_episode_steps, env_spec.reward_threshold



def play(
              name, 
              learner,
              nodes_per_layer: int,
              layer_count,
              learning_rate: float,
              random_choice_min_rate: float,
              sample_size: int,
              verbose: float = 1,
              max_episodes: int = 9999999,
              max_steps: int = 40000000,
              name_prefix="",
              experience_creator=Experience,
              buffer_creator=ReplayBuffer,
              data_func=convert_atari_frame,
              window: int = 4,
              target_network_interval=None,
              start_length=200000,
              end_length=1000000,
              random_decay_end=4000000,  # Decay rate used for DQN in Rainbow.
              *args, **kwargs):

    # Seed random variables
    #np.random.seed(4)
    #random.seed(4)  # TODO May not be needed

    env_name, max_episode_steps, reward_threshold = get_env_info(name)
    env = gym.make(env_name)
    if verbose > 2:
        # https://github.com/openai/gym/wiki/FAQ
        env = gym.wrappers.Monitor(env, '.videos/' + str(time()) + '/')



    if len(env.observation_space.shape) == 1:
        feature_count = env.observation_space.shape[0]
    else:
        feature_count = np.array(env.observation_space.shape)
        print(f"features: {feature_count}")
        feature_count = (*feature_count[0:-1], window)
    action_count = env.action_space.n

    # Scale gamma to approach zero near max_episode_steps
    #gamma = float(np.power(0.0001, 1. / max_episode_steps))

    learner.build_model(input_dimension=feature_count, output_dimension=action_count,
                        nodes_per_layer=nodes_per_layer,
                        learning_rate=learning_rate,
                        layer_count=layer_count,
                        *args, **kwargs)

    # TODO account for possible extra space from scoring

    agent = Agent(
        learner=learner,
        sample_size=sample_size,
        replay_buffer=buffer_creator(max_length=end_length, start_length=start_length),
        environment=env,
        reward_threshold=reward_threshold,
        random_choice_decay_min=random_choice_min_rate,
        max_episode_steps=max_episode_steps,
        max_episodes=max_episodes,
        early_stopping=False,
        verbose=verbose,
        seed=4,
        experience_creator=experience_creator,
        observation_processor=data_func,
        window=window,
        target_network_interval=target_network_interval,
        random_decay_end=random_decay_end,
        name_prefix=name_prefix)
    best_score, rolling_average_score, steps = agent.play(max_steps, verbose=verbose)
    #best_score, rolling_average_score, steps = agent.play(max_episodes * max_episode_steps, verbose=verbose)
    #score = agent.score_model(100, verbose=0)

    return best_score, rolling_average_score, steps

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="""
    This script will train a Deep Q Agent 
    """)
    
    # Play args
    parser.add_argument("--environment", default="Breakout-v4", help="")
    parser.add_argument("--random_choice_min_rate", default=0.01, help="number of dense nodes for encoder")
    parser.add_argument("--sample_size", default=32, help="Number of epochs to run")
    parser.add_argument("--verbose", default=1, help="None")
    parser.add_argument("--max_episodes", default=999999999, help="None")
    parser.add_argument("--max_steps", default=40000000, help="None")
    parser.add_argument("--name_prefix", default='', help="None")
    parser.add_argument("--experience_creator", default=Experience, help="None")
    parser.add_argument("--buffer_creator", default=ReplayBuffer, help="None")
    #parser.add_argument("--data_func", default=None, help="None")
    parser.add_argument("--window", default=4, help="None")
    parser.add_argument("--target_network_interval", default=  10000, help="None")
    parser.add_argument("--start_length",            default= 200000, help="None")
    parser.add_argument("--end_length",              default=1000000, help="None")
    parser.add_argument("--random_decay_end",        default=4000000, help="None")
    
    # Learner Args
    parser.add_argument("--learner_type", default="clipped_double_duel", help="")
    parser.add_argument("--nodes_per_layer", default=512, help="")
    parser.add_argument("--layer_count", default=1, help="")
    parser.add_argument("--gamma", default=0.99, help="None")
    parser.add_argument("--learning_rate", default=0.00001, help="None")
    parser.add_argument("--conv_nodes", default=[32, 64, 64], help="None")
    parser.add_argument("--kernel_size", default=[8, 4, 3], help="None")
    parser.add_argument("--conv_stride", default=[4, 2, 1], help="None")

    args = parser.parse_args()


    if args.learner_type == "vanilla":
        #learner = DeepQFactory.create_vanilla_deep_q()
        pass
    elif args.learner_type == "double":
        #learner = DeepQFactory.create_double_deep_q()
        pass
    elif args.learner_type == "clipped_double":
        #learner = DeepQFactory.create_clipped_double_deep_q()
        pass
    elif args.learner_type == "clipped_double_duel":
        # TODO only one that works currently
        learner = DeepQFactory.create_atari_clipped_double_duel_deep_q()
    elif args.learner_type == "duel":
        #learner = DeepQFactory.create_double_duel_deep_q()
        pass
    elif args.learner_type == "double_duel":
        #learner = DeepQFactory.create_double_duel_deep_q()
        pass
    else:
        raise ValueError("Invalid learner_type")

    print(args)
    play(
        name=args.environment,
        learner=learner,
        nodes_per_layer=args.nodes_per_layer,
        layer_count=args.layer_count,
        learning_rate=args.learning_rate,
        random_choice_min_rate=args.random_choice_min_rate,
        sample_size=args.sample_size,
        verbose=args.verbose,
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        name_prefix=args.name_prefix,
        experience_creator=args.experience_creator,
        buffer_creator=args.buffer_creator,
        #data_func=args.data_func,
        window=args.window,
        target_network_interval=args.target_network_interval,
        start_length=args.start_length,
        end_length=args.end_length,
        random_decay_end=args.random_decay_end,
        conv_nodes=args.conv_nodes,
        kernel_size=args.kernel_size,
        conv_stride=args.conv_stride,
        double_deep_q=True, is_dueling=True, clipped_double_deep_q=True
    )


    # agent.learner.model.load_weights("model.h5")
    #agent.learner.model.save_weights("model.h5")