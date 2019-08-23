import gym
import gymfc

import numpy as np
from datetime import datetime
import os
from shutil import copyfile

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import TRPO

# Preliminary parameter definition
TEST_STEPS = 2000
TRAINING_INTERVAL_STEPS = 10000
TOTAL_TRAINING_STEPS = 1e12
RESULTS_PATH = "/mnt/Data_Ubuntu/results_training/" + datetime.now().strftime("%B-%d-%Y_%H_%M%p")
TRAINING_NAME = "ppo2"
AGENT_ALGORITHM = "DDPG" # DDPG, PPO2, TRPO
PLOTTING_INFORMATION = False

# PRETRAINED_MODEL ="/mnt/Data_Ubuntu/results_training/August-21-2019_08_12AM_ppo2/ppo2_0000220000.pkl"
PRETRAINED_MODEL = None
TEST_ONLY = False

if (PLOTTING_INFORMATION == True):
    import rospy
    from std_msgs.msg import Float32MultiArray

# Create environment
env = gym.make('CaRL_GymFC-MotorVel_M4_Ep-v0')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

# Create global experiments path
if not TEST_ONLY:
    global_path = RESULTS_PATH + "_" + TRAINING_NAME + "/"
else:
    global_path = RESULTS_PATH + "_" + TRAINING_NAME + "_test" + "/"

os.makedirs(global_path, exist_ok=True)

# Copy file to results directory
if PRETRAINED_MODEL:
    pretrained_model_name = "pretrained.pkl"
    copyfile(PRETRAINED_MODEL, global_path + pretrained_model_name)

# Define model
if AGENT_ALGORITHM == "DDPG":
    # Add some param noise for exploration
    param_noise = None
    action_noise = None
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01)
    # n_actions = env.action_space.shape[-1]
    # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    # Because we use parameter noise, we should use a MlpPolicy with layer normalization
    model = DDPG(LnMlpPolicy, env, param_noise=param_noise, action_noise=action_noise, verbose=1, tensorboard_log=global_path + "tb")

    # Load if pretrained
    if PRETRAINED_MODEL:
        del model
        model = DDPG.load(global_path + pretrained_model_name, env=env)
        print("INFO: Loaded model " + global_path + pretrained_model_name)

elif AGENT_ALGORITHM == "PPO2":
    # Create model
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=global_path + "tb", cliprange=0.075)

    # Load if pretrained
    if PRETRAINED_MODEL:
        del model
        model = PPO2.load(global_path + pretrained_model_name, env=env,tensorboard_log=global_path + "tb",cliprange=0.1)
        print("INFO: Loaded model " + global_path + pretrained_model_name)

elif AGENT_ALGORITHM == "TRPO":
    # Create model
    model = TRPO(MlpPolicy, env, verbose=1, tensorboard_log=global_path + "tb")

    # Load if pretrained
    if PRETRAINED_MODEL:
        del model
        model = TRPO.load(global_path + pretrained_model_name, env=env)
        print("INFO: Loaded model " + global_path + pretrained_model_name)
else:
    raise RuntimeError('ERROR: Agent not recognized')


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
            msg.data = info[0]['forwarded_action']
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

    if not TEST_ONLY:
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

    if not TEST_ONLY:
        # Saving model
        print("Saving in '" + global_path + "'")
        model.save(global_path + TRAINING_NAME + "_" + str(int(t)).zfill(10))

    # Update t
    t = t + TRAINING_INTERVAL_STEPS

env.close()
