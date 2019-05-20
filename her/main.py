import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import argparse
import pprint as pp
import gym_hypercube
import pickle
import os

from model import ActorNetwork, CriticNetwork
from noise import OrnsteinUhlenbeckActionNoise
from train import train


def main(args):

    with tf.Session() as sess:
        env = gym.make(args['env'])
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        # assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim, float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']), actor.get_num_trainable_vars())

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)
        if args['train']:
            train(sess, env, args, actor, critic, actor_noise)
        else:
            ddpg = []
            indexes = [e for e in range(500) if e % 10 == 9]
            indexes = [0] + indexes
            num_test_tasks = 100
            directory = 'ddpg_only_no_gcp'
            for index in indexes:
                print(index)
                times = []
                saver = tf.train.Saver()
                saver.restore(sess, "../meta_rl/models_01/{0}/model-{1}.ckpt".format(directory, index))
                tasks = env.unwrapped.sample_tasks(num_test_tasks)
                success = 0
                for task in tasks:
                    s = env.reset_task(task)
                    step = 0
                    d = False
                    while not d:
                        # env.render()
                        action = actor.predict_target(np.reshape(s, (1, actor.s_dim)))[0]
                        step += 1
                        obs, r, d, _ = env.step(action)
                    # if r == 1:
                    #    success += 1
                    env.close()
                    times.append(step)
                # ddpg.append(success / num_test_tasks)
                ddpg.append(times)
            with open('../ddpg_only_time_D.pkl', 'wb') as f:
                pickle.dump(ddpg, f)
            with open('../ddpg_only_time_D.pkl', 'rb') as f:
                test = pickle.load(f)
            print(test[0])


if __name__ == '__main__':
    id = gym_hypercube.dynamic_register(n_dimensions=2,
                                        env_description={'high_reward_value': 1,
                                                         'low_reward_value': 0,
                                                         'nb_target': 1,
                                                         'mode': 'random',
                                                         'agent_starting': 'random',
                                                         'generation_zone': 'abc',
                                                         'speed_limit_mode': 'vector_norm',
                                                         'GCP': True},
                                        continuous=True,
                                        acceleration=True,
                                        reset_radius=None)
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.90)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on gym_hypercube', default=id)
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1236)
    parser.add_argument('--epochs', help='number of epochs', default=20)
    parser.add_argument('--max-episodes', help='max num of episodes per epoch to do while training', default=20)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=201)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/del')
    parser.add_argument('--HER', help='use hindsight experience replay', default=False)
    parser.add_argument('--train', help='train the model from scratch', default=True)

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)

    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
