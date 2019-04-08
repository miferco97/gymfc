import math
import numpy as np
from .gazebo_env import GazeboEnv
import logging
logger = logging.getLogger("gymfc")
from math import pi


import numpy as np


import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation



# This function is called periodically from FuncAnimation


class AttitudeFlightControlEnv(GazeboEnv):
    def __init__(self, **kwargs):
        self.last_angular_part = 0;
        self.last_theta_norm=0;
        self.error_raw=np.zeros(7)
        self.index = 0
        self.max_sim_time = kwargs["max_sim_time"]

        np.random.seed(seed=None)
        super(AttitudeFlightControlEnv, self).__init__()

    def compute_reward(self):
        rew_R = np.abs(self.obs.euler[0]/(pi/2))
        rew_P = np.abs(self.obs.euler[1]/(pi/2))
        rew_Y = np.abs(self.obs.euler[2]/pi)
        # if
        #
        # print("reward R: ", rew_R)
        # print("reward P: ", rew_P)
        # print("reward Y: ", rew_Y)

        # reward = - (rew_P+rew_R+rew_Y) / 3
        reward = - (rew_P)
        # print("total reward",reward)
        # print("-----------")

        return reward

    def compute_reward_1(self):

        ref_quat = np.asarray([1,0,0,0])
        actual_quat = self.obs.orientation_quat
        # norm_cuat = np.sqrt(np.sum(np.power(self.obs.orientation_quat,2),axis=0))
        # print("norm_cuat",norm_cuat)
        # print("actual cuat",actual_quat)
        theta=2*np.arccos(np.dot(ref_quat,actual_quat))


        # actual_theta_norm = theta/(2*pi)
        actual_theta_norm = (np.exp(theta/(2*pi)) - 1) /(np.exp(1)-1 + 1e-12)
        action_part = (np.sum((1 + self.last_action) / 2) / 4)  # action part between [0 y 1]

        reward = - actual_theta_norm #- 0.001 * action_part
        #reward = self.last_theta_norm - actual_theta_norm
        # self.last_theta_norm = actual_theta_norm

        return reward

        # return -np.clip(np.sum(np.abs(self.error)) / (self.omega_bounds[1] * 3), 0, 1)


    # def compute_reward(self):
    #     """ Compute the reward """
    #
    #     # self.angular_part = - np.sum(np.abs(self.error)) / 3
    #
    #     self.angular_part = - math.exp(np.sum(np.abs(self.error)))
    #
    #     reward = self.angular_part - self.last_angular_part #+ action_part
    #     self.last_angular_part =self.angular_part
    #
    #     #action_part = - 0.01 * (np.sum((1 + self.last_action)/2)/4) # action part between [0 y 1]
    #     return reward

        # return -np.clip(np.sum(np.abs(self.error)) / (self.omega_bounds[1] * 3), 0, 1)

    def sample_target(self):
        """ sample a random angle """
        return np.asfarray([0,0,0])



class CaRL_env(AttitudeFlightControlEnv):
    def __init__(self, **kwargs):

        self.observation_history = []
        self.memory_size = kwargs["memory_size"]
        super(CaRL_env, self).__init__(**kwargs)
        self.omega_target = self.sample_target()
        self.last_reward=0;
        self.render()

        self.fig, self.ax = plt.subplots(2,sharex=True)
        # self.ax = self.fig.add_subplot(211)
        self.xs = []
        self.ys = []
        self.x1 = []
        self.x2 = []
        self.x3 = []
        self.x4 = []

    def animate(self):
        # Read temperature (Celsius) from TMP102

        self.index = self.index + 1
        # print(self.index)
        if self.index >= 100:

            # Add x and y to lists
            self.xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
            self.ys.append(self.last_reward)

            self.x1.append(self.last_action[0])
            self.x2.append(self.last_action[1])
            self.x3.append(self.last_action[2])
            self.x4.append(self.last_action[3])

            # print("rew: ", self.last_reward)

            # Limit x and y lists to 20 items
            self.xs = self.xs[-20:]
            self.ys = self.ys[-20:]

            self.x1 = self.x1[-20:]
            self.x2 = self.x2[-20:]
            self.x3 = self.x3[-20:]
            self.x4 = self.x4[-20:]

            # Draw x and y lists
            self.ax[0].clear()
            self.ax[1].clear()

            self.ax[0].plot(self.xs, self.ys )
            plt.title('Reward')
            plt.ylabel('Reward value')

            self.ax[1].plot(self.xs, self.x1 ,'r',self.xs, self.x2 ,'g',self.xs, self.x3 ,'b',self.xs, self.x4 ,'k' )
            plt.ylabel('Actions value')
            plt.xlabel('Time')
            # Format plot
            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.30)
            plt.show(block=False)
            plt.pause(0.001)
            self.index = 0


# Set up plot to call animate() function periodically


    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # print("action:", action)
        # Step the sim

        self.last_action = action;

        self.obs = self.step_sim(action)

        quat = self.obs.orientation_quat
        self.speeds = self.obs.angular_velocity_rpy / (self.omega_bounds[1])

        state = np.append(quat, self.speeds)
        self.observation_history.append(np.concatenate([state, self.obs.motor_velocity]))

        reward = self.compute_reward()
        self.last_reward = reward;
        # self.animate()

        # print("pitch:", self.obs.euler[1])
        if self.sim_time >= self.max_sim_time:
            done = True
            self.last_theta_norm=0
        # elif np.abs(self.obs.euler[2]) >= pi / 2 or np.abs(self.obs.euler[1]) >= 0.99 * (pi / 2):
        elif np.abs(self.obs.euler[0]) >= 0.999 * (pi / 2) or  np.abs(self.obs.euler[1]) >= 0.999 * (pi / 2):
            done = True
        #     self.last_angular_part=0;
            reward = -1000 * (self.max_sim_time-self.sim_time)
        else:
            done = False

        info = {"sim_time": self.sim_time, "sp": self.omega_target, "current_rpy": self.omega_actual}
        # print("state: ",state);

        return state, reward, done, info

    def state(self):
        """ Get the current state """


        return np.zeros(7)

    def reset(self):
        self.observation_history = []
        return super(CaRL_env, self).reset()



class GyroErrorFeedbackEnv(AttitudeFlightControlEnv):
    def __init__(self, **kwargs): 
        self.observation_history = []
        self.memory_size = kwargs["memory_size"]
        super(GyroErrorFeedbackEnv, self).__init__(**kwargs)
        self.omega_target = self.sample_target()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high) 
        # Step the sim
        self.obs = self.step_sim(action) # all observations
        self.error = (self.omega_target - self.obs.angular_velocity_rpy)
        self.observation_history.append(np.concatenate([self.error]))
        state = self.state()
        done = self.sim_time >= self.max_sim_time
        reward = self.compute_reward()
        info = {"sim_time": self.sim_time, "sp": self.omega_target, "current_rpy": self.omega_actual}
        return state, reward, done, info

    def state(self):
        """ Get the current state """
        # The newest will be at the end of the array
        memory = np.array(self.observation_history[-self.memory_size:])
        return np.pad(memory.ravel(), 
                      ( (3 * self.memory_size) - memory.size, 0), 
                      'constant', constant_values=(0)) 

    def reset(self):
        self.observation_history = []
        return super(GyroErrorFeedbackEnv, self).reset()

class GyroErrorESCVelocityFeedbackEnv(AttitudeFlightControlEnv):
    def __init__(self, **kwargs): 
        self.observation_history = []
        self.memory_size = kwargs["memory_size"]
        super(GyroErrorESCVelocityFeedbackEnv, self).__init__(**kwargs)
        self.omega_target = self.sample_target()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high) 
        # Step the sim
        self.obs = self.step_sim(action)
        self.error = self.omega_target - self.obs.angular_velocity_rpy
        self.observation_history.append(np.concatenate([self.error, self.obs.motor_velocity]))
        state = self.state()
        done = self.sim_time >= self.max_sim_time
        reward = self.compute_reward()
        info = {"sim_time": self.sim_time, "sp": self.omega_target, "current_rpy": self.omega_actual}

        return state, reward, done, info

    def state(self):
        """ Get the current state """
        # The newest will be at the end of the array
        memory = np.array(self.observation_history[-self.memory_size:])
        return np.pad(memory.ravel(), 
                      (( (3+self.motor_count) * self.memory_size) - memory.size, 0), 
                      'constant', constant_values=(0)) 

    def reset(self):
        self.observation_history = []
        return super(GyroErrorESCVelocityFeedbackEnv, self).reset()

class GyroErrorESCVelocityFeedbackContinuousEnv(GyroErrorESCVelocityFeedbackEnv):
    def __init__(self, **kwargs): 
        self.command_time_off = kwargs["command_time_off"]
        self.command_time_on = kwargs["command_time_on"]
        self.command_off_time = None
        super(GyroErrorESCVelocityFeedbackContinuousEnv, self).__init__(**kwargs)

    def step(self, action):
        """ Sample a random angular velocity """
        ret = super(GyroErrorESCVelocityFeedbackContinuousEnv, self).step(action) 

        # Update the target angular velocity 
        if not self.command_off_time:
            self.command_off_time = self.np_random.uniform(*self.command_time_on)
        elif self.sim_time >= self.command_off_time: # Issue new command
            # Commands are executed as pulses, always returning to center
            if (self.omega_target == np.zeros(3)).all():
                self.omega_target = self.sample_target() 
                self.command_off_time = self.sim_time  + self.np_random.uniform(*self.command_time_on)
            else:
                self.omega_target = np.zeros(3)
                self.command_off_time = self.sim_time  + self.np_random.uniform(*self.command_time_off) 

        return ret 


