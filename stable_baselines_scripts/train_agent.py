import gym
import gymfc

import numpy as np
from datetime import datetime
import os

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import DDPG
from stable_baselines.ddpg import AdaptiveParamNoiseSpec

# Preliminary parameter definition
TEST_STEPS = 2000
TRAINING_INTERVAL_STEPS = 20000
TOTAL_TRAINING_STEPS = 1e12
RESULTS_PATH = "/home/alejo/py_workspace/stable-baselines/results/" + datetime.now().strftime("%B-%d-%Y_%H_%M%p")
TRAINING_NAME = "ddpg_gymfc"
PLOTTING_INFORMATION = True

if (PLOTTING_INFORMATION == True):
    import rospy
    from std_msgs.msg import Float32MultiArray

# Create environment
env = gym.make('CaRL_GymFC-MotorVel_M4_Ep-v0')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

# Create global experiments path
global_path = RESULTS_PATH + "_" + TRAINING_NAME + "/"
os.makedirs(global_path, exist_ok=True)

# Define model
# Add some param noise for exploration
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
# Because we use parameter noise, we should use a MlpPolicy with layer normalization
model = DDPG(LnMlpPolicy, env, param_noise=param_noise, verbose=1, tensorboard_log=global_path + "tb")
#model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=global_path + "tb")

def evaluate(model, num_steps=1000, pub=None):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """

    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(action)

        if (pub != None):
            # Publish action
            msg = Float32MultiArray()
            msg.data = action[0]
            pub.publish(msg)

        # Stats
        episode_rewards[-1] += rewards[0]
        if dones[0]:
            obs = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward

# Step counter initialization
t = 0

if (PLOTTING_INFORMATION == True):
    # Ros publisher
    pub = rospy.Publisher('agent_actions', Float32MultiArray, queue_size=10)
    rospy.init_node('agent', anonymous=True)

# Main loop
while(t < TOTAL_TRAINING_STEPS):
    # Train model
    if (t == 0):
        model.learn(total_timesteps=TRAINING_INTERVAL_STEPS)
    else:
        model.learn(total_timesteps=TRAINING_INTERVAL_STEPS, reset_num_timesteps=False)

    # Evaluate model
    print("Testing model...")
    if (PLOTTING_INFORMATION == True):
        evaluate(model, num_steps=TEST_STEPS, pub=pub)
    else:
        evaluate(model, num_steps=TEST_STEPS)

    # Saving model
    print("Saving in '" + global_path + "'")
    model.save(global_path + TRAINING_NAME + "_" + str(int(t / TRAINING_INTERVAL_STEPS)).zfill(4))

    # Update t
    t = t + TRAINING_INTERVAL_STEPS

env.close()