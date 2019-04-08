
import gymfc
import gym
import tensorflow as tf
import numpy as np
import time

env = gym.make('CaRL_GymFC-MotorVel_M4_Ep-v0')
env.reset()
# env.render()
# print(env.observation_space.shape)
for i in range(100):

    for i in range(100000000):

        action = env.action_space.sample()
       #  action = np.asarray([0 ,0 ,0,0])
        state, reward, done, info = env.step(action)
        # print(env.action_space.low)
        print("state: ", state)
        # print("velocidad: ", state[0:3])
        #print("angulo: ", state[3:6])
        # print("ref: ", state[6:10])
        #print(action)
        # print(env.observation_space.shape)
        print("reward: ", reward)
        # print(env.omega_target)
    env.reset()
env.close()

