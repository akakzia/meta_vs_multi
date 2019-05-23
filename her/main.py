import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import argparse
import pprint as pp
import gym_hypercube
import pickle
import os
import json

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

        if args['train']:
            if not os.path.exists(args['save_dir']):
                os.makedirs(args['save_dir'])
            with open(os.path.join(args['save_dir'], 'config.json'), 'w') as f:
                json.dump(args, f, indent=2)
            train(sess, env, args, actor, critic, actor_noise)
        else:
            ddpg = []
            indexes = [e for e in range(500) if e % 10 == 9]
            indexes = [0] + indexes
            num_test_tasks = 10
            successes = []
            directory = args['to_pickle']
            for index in indexes:
                times = []
                saver = tf.train.Saver()
                saver.restore(sess, "../models/{0}/model-{1}.ckpt".format(directory, index))
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
                    if r == 1:
                        success += 1
                    times.append(step)
                env.close()
                successes.append(success / num_test_tasks)
                ddpg.append(times)
            out = [successes, ddpg]
            #if not os.path.exists('./pkls'):
            #    os.makedirs('./pkls')
            #with open('./pkls/{0}.pkl'.format(args['save_dir']), 'wb') as f:
            #    pickle.dump(out, f)
            #with open('./pkls/{0}.pkl'.format(args['save_dir']), 'rb') as f:
            #    test = pickle.load(f)


if __name__ == '__main__':
    id = gym_hypercube.dynamic_register(n_dimensions=2,
                                        env_description={'high_reward_value': 1,
                                                         'low_reward_value': 0,
                                                         'nb_target': 1,
                                                         'mode': 'random',
                                                         'agent_starting': 'fixed',
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
    parser.add_argument('--epochs', help='number of epochs', default=500)
    parser.add_argument('--max-episodes', help='max num of episodes per epoch to do while training', default=30)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=201)
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./summary/del')
    parser.add_argument('--save-dir', help='directory for storing models', default='del')
    parser.add_argument('--to-pickle', help='model to pickle', default='all_fictive_DDPG+HER')
    parser.add_argument('--HER', help='use hindsight experience replay', default=False)
    parser.add_argument('--fictive-rewards', help='use hindsight experience replay', default='all')
    parser.add_argument('--train', help='train the model from scratch', default=False)

    args = vars(parser.parse_args())

    if args['HER'] == 'True':
        args['HER'] = True
    if args['HER'] == 'False':
        args['HER'] = False

    pp.pprint(args)

    main(args)
