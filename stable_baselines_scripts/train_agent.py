import gym
import gymfc

import numpy as np
from datetime import datetime
import os
import os.path
from shutil import copyfile
import yaml
import sys

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import TRPO
from stable_baselines import POSITION_PID_REAL
from stable_baselines import POS_VEL_PID_REAL
from stable_baselines import POSITION_PID_SIM
from stable_baselines import POS_VEL_PID_SIM

class ConfigLoadException(Exception):
    pass

if len(sys.argv) < 2:
    message = "YAML config file not provided"
    raise ConfigLoadException(message)
else:
    config_file = sys.argv[1]

if "GYMFC_CONFIG" not in os.environ:
    message = (
            "Environment variable {} not set. " +
            "Before running the environment please execute, " +
            "'export {}=path/to/config/file' " +
            "or add the variable to your .bashrc."
    ).format("GYMFC_CONFIG", "GYMFC_CONFIG")
    raise ConfigLoadException(message)

config_path = os.environ["GYMFC_CONFIG"]
config_path = config_path.rsplit('/', 1)[0]

# Read YAML parameters
with open(os.path.join(config_path, config_file), 'r') as stream:
    data_loaded = yaml.load(stream)

# Print information
print("------ Configuration parameters ------")
print(data_loaded)
print("--------------------------------------")

# Parameter reading
TEST_STEPS = int(data_loaded['conf']['test_steps'])
TRAINING_INTERVAL_STEPS = int(data_loaded['conf']['training_interval_steps'])
TOTAL_TRAINING_STEPS = int(float(data_loaded['conf']['total_training_steps']))
RESULTS_PATH = os.environ["HOME"] + "/" + data_loaded['conf']['results_path'] + datetime.now().strftime("%B-%d-%Y_%H_%M%p")
TRAINING_NAME = data_loaded['conf']['training_name']
AGENT_ALGORITHM = data_loaded['conf']['agent_algorithm'] # DDPG, PPO2, TRPO, POSITION_PID, POS_VEL_PID
PLOTTING_INFORMATION = bool(int(data_loaded['conf']['plotting_information']))
PRETRAINED_MODEL = os.environ["HOME"] + "/" + data_loaded['conf']['pretrained_model']
TEST_ONLY = bool(int(data_loaded['conf']['test_only']))

if PLOTTING_INFORMATION:
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
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=global_path + "tb", cliprange=0.1)

    # Load if pretrained
    if PRETRAINED_MODEL:
        del model
        model = PPO2.load(global_path + pretrained_model_name, env=env)
        print("INFO: Loaded model " + global_path + pretrained_model_name)

elif AGENT_ALGORITHM == "TRPO":
    # Create model
    model = TRPO(MlpPolicy, env, verbose=1, tensorboard_log=global_path + "tb")

    # Load if pretrained
    if PRETRAINED_MODEL:
        del model
        model = TRPO.load(global_path + pretrained_model_name, env=env)
        print("INFO: Loaded model " + global_path + pretrained_model_name)

elif AGENT_ALGORITHM == "POSITION_PID_REAL":
    try:
        # Create model
        model = POSITION_PID_REAL(env,  float(data_loaded['POSITION_PID_REAL']['kp_R']),
                                    float(data_loaded['POSITION_PID_REAL']['ki_R']),
                                    float(data_loaded['POSITION_PID_REAL']['kd_R']),
                                     float(data_loaded['POSITION_PID_REAL']['kp_P']),
                                     float(data_loaded['POSITION_PID_REAL']['ki_P']),
                                     float(data_loaded['POSITION_PID_REAL']['kd_P']),
                                     float(data_loaded['POSITION_PID_REAL']['kp_Y']),
                                     float(data_loaded['POSITION_PID_REAL']['ki_Y']),
                                     float(data_loaded['POSITION_PID_REAL']['kd_Y']))
    except:
        # Print warning info
        print("WARN: YAML configuration not found for this algorithm")

        # Create model
        model = POSITION_PID_REAL(env,  13.0, 0.0, 10.0,
                                   13.0, 0.0, 10.0,
                                   1.5, 0.0,  10.0)

elif AGENT_ALGORITHM == "POS_VEL_PID_REAL":
    try:
        # Create model
        model = POS_VEL_PID_REAL(env,   float(data_loaded['POS_VEL_PID_REAL']['kp_R_pos']),
                                float(data_loaded['POS_VEL_PID_REAL']['kp_P_pos']),
                                float(data_loaded['POS_VEL_PID_REAL']['kp_Y_pos']),
                                float(data_loaded['POS_VEL_PID_REAL']['kp_R_vel']),
                                float(data_loaded['POS_VEL_PID_REAL']['ki_R_vel']),
                                float(data_loaded['POS_VEL_PID_REAL']['kd_R_vel']),
                                float(data_loaded['POS_VEL_PID_REAL']['kp_P_vel']),
                                float(data_loaded['POS_VEL_PID_REAL']['ki_P_vel']),
                                float(data_loaded['POS_VEL_PID_REAL']['kd_P_vel']),
                                float(data_loaded['POS_VEL_PID_REAL']['kp_Y_vel']),
                                float(data_loaded['POS_VEL_PID_REAL']['ki_Y_vel']),
                                float(data_loaded['POS_VEL_PID_REAL']['kd_Y_vel']))
    except:
        # Print warning info
        print("WARN: YAML configuration not found for this algorithm")

        # Create model
        model = POS_VEL_PID_REAL(env,   6.0, 8.0, 1.5,
                                   0.003, 0.0002, 0.004,
                                   0.003, 0.0002, 0.004,
                                   0.003, 0.0002, 0.004)

elif AGENT_ALGORITHM == "POSITION_PID_SIM":
    try:
        # Create model
        model = POSITION_PID_SIM(env,  float(data_loaded['POSITION_PID_SIM']['kp_R']),
                                    float(data_loaded['POSITION_PID_SIM']['ki_R']),
                                    float(data_loaded['POSITION_PID_SIM']['kd_R']),
                                     float(data_loaded['POSITION_PID_SIM']['kp_P']),
                                     float(data_loaded['POSITION_PID_SIM']['ki_P']),
                                     float(data_loaded['POSITION_PID_SIM']['kd_P']),
                                     float(data_loaded['POSITION_PID_SIM']['kp_Y']),
                                     float(data_loaded['POSITION_PID_SIM']['ki_Y']),
                                     float(data_loaded['POSITION_PID_SIM']['kd_Y']))
    except:
        # Print warning info
        print("WARN: YAML configuration not found for this algorithm")

        # Create model
        model = POSITION_PID_SIM(env,  13.0, 0.0, 10.0,
                                   13.0, 0.0, 10.0,
                                   1.5, 0.0,  10.0)

elif AGENT_ALGORITHM == "POS_VEL_PID_SIM":
    try:
        # Create model
        model = POS_VEL_PID_SIM(env,   float(data_loaded['POS_VEL_PID_SIM']['kp_R_pos']),
                                float(data_loaded['POS_VEL_PID_SIM']['kp_P_pos']),
                                float(data_loaded['POS_VEL_PID_SIM']['kp_Y_pos']),
                                float(data_loaded['POS_VEL_PID_SIM']['kp_R_vel']),
                                float(data_loaded['POS_VEL_PID_SIM']['ki_R_vel']),
                                float(data_loaded['POS_VEL_PID_SIM']['kd_R_vel']),
                                float(data_loaded['POS_VEL_PID_SIM']['kp_P_vel']),
                                float(data_loaded['POS_VEL_PID_SIM']['ki_P_vel']),
                                float(data_loaded['POS_VEL_PID_SIM']['kd_P_vel']),
                                float(data_loaded['POS_VEL_PID_SIM']['kp_Y_vel']),
                                float(data_loaded['POS_VEL_PID_SIM']['ki_Y_vel']),
                                float(data_loaded['POS_VEL_PID_SIM']['kd_Y_vel']))
    except:
        # Print warning info
        print("WARN: YAML configuration not found for this algorithm")

        # Create model
        model = POS_VEL_PID_SIM(env,   1.5, 1.5, 0.05,
                                   20.0, 0.0, 0.0,
                                   20.0, 0.0, 0.0,
                                   7.0, 0.0, 0.0)

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
            print("Reseting")
            obs = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward

# Step counter initialization
t = 0

if PLOTTING_INFORMATION == True:
    # Ros publisher
    pub = rospy.Publisher('agent_actions', Float32MultiArray, queue_size=10)
    try:
        rospy.init_node('agent', anonymous=True)
    except:
        print("WARN: ROS node already initialized")

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
