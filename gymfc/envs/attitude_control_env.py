import math
import numpy as np
from .gazebo_env import GazeboEnv
import logging
logger = logging.getLogger("gymfc")
from math import pi

import time, threading
import asyncio
import nest_asyncio
nest_asyncio.apply()


import numpy as np


import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# This function is called periodically from FuncAnimation


class AttitudeFlightControlEnv(GazeboEnv):
    def __init__(self, **kwargs):
        self.last_angular_part = 0
        self.last_theta_norm=0
        self.error_raw=np.zeros(7)
        self.random_quaternion=np.zeros(4)
        self.random_euler=np.zeros(3)
        self.index = 0
        self.max_sim_time = kwargs["max_sim_time"]
        self.incr_action = []
        self.last_action = []
        self.action = []

        np.random.seed(seed=None)
        super(AttitudeFlightControlEnv, self).__init__()


    def get_random_quat(self):

        pitch = 0.9 * (pi * np.random.random_sample() - pi / 2)
        roll = 0.9 * (pi * np.random.random_sample() - pi / 2)
        yaw = 0.9 * (2*pi * np.random.random_sample() - pi)

        self.random_euler[0] = roll
        self.random_euler[1] = pitch
        self.random_euler[2] = yaw

        # self.random_euler[0] = 0.3
        # self.random_euler[1] = 0
        # self.random_euler[2] = 0


        print (self.random_euler[0],self.random_euler[1],self.random_euler[2])
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        self.random_quaternion[0] = cy * cp * cr + sy * sp * sr
        self.random_quaternion[1] = cy * cp * sr - sy * sp * cr
        self.random_quaternion[2] = sy * cp * sr + cy * sp * cr
        self.random_quaternion[3] = sy * cp * cr - cy * sp * sr

    def compute_reward_ExpA(self):
        rew_R = np.abs((self.obs.euler[0])/(pi))
        rew_P = np.abs((self.obs.euler[1])/(pi))
        rew_Y = np.abs((self.obs.euler[2])/(pi))

        if np.abs(self.obs.euler[0]) >= 0.999 * (pi / 2) or np.abs(self.obs.euler[1]) >= 0.999 * (pi / 2):
            done = True
            reward = -1
        else:
            reward = np.power(1 - np.clip(((rew_P + rew_R + rew_Y) / 3),0,1),3)
            done = False

        return [reward, done]

    def compute_reward_ExpB(self):
        # Init action reward
        action_reward = 0
        shap_incr_act = 0

        # Compute shaping
        shap_R = -100 * np.sqrt(np.power(self.obs.euler[0] / pi, 2))
        shap_P = -100 * np.sqrt(np.power(self.obs.euler[1] / pi, 2))
        shap_Y = -100 * np.sqrt(np.power(self.obs.euler[2] / pi, 2))

        # print("Shaping pitch: " + str(shap_P))

        if np.asarray(self.incr_action).size:
            shap_incr_act = -1 * np.sqrt(np.power(np.clip(np.sum(np.abs(self.incr_action)) / np.asarray(self.incr_action).size, a_min=0, a_max=1), 2))
            # print("Shaping action: " + str(shap_incr_act))
        reward = shap_R + shap_P + shap_Y + shap_incr_act

        # if np.asarray(self.incr_action).size:
        #     action_reward = np.clip(np.sum(np.abs(self.incr_action)) / np.asarray(self.incr_action).size, a_min=0, a_max=1)
        #     print("Action reward: " + str(action_reward))

        # Compute shaping
        if self.last_reward != 0:
            reward = reward - self.last_reward
            self.last_reward = shap_R + shap_P + shap_Y
            # print("Shaped reward: " + str(reward))
        else:
            self.last_reward = reward
            reward = 0

        # Include action velocity penalization
        reward += action_reward

        if np.abs(self.obs.euler[0]) >= 0.999 * (pi / 2) or np.abs(self.obs.euler[1]) >= 0.999 * (pi / 2):
            done = True
            reward = -50
        else:
            done = False

        return [reward, done]

    def compute_reward(self):

        # Compute reward
        [reward, done] = self.compute_reward_ExpA()
        # [reward, done] = self.compute_reward_ExpB()

        # Store last reward
        if done:
            self.last_reward = 0
            self.last_action = []
            self.action = []

        return [reward, done]



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
        self.last_reward = 0
        self.render()
        self.start = 0
        self.FREQUENCY = 70.0
        self.THREAD_PERIOD = (1.0 / (10.0 * self.FREQUENCY)) # 10 times faster
        self.th_counter = 0
        self.pivot = False
        self.paused = False
        self.lock = threading.RLock()
        self.start_sim = 0

        self.fig, self.ax = plt.subplots(2,sharex=True)
        # self.ax = self.fig.add_subplot(211)
        self.xs = []
        self.ys = []
        self.x1 = []
        self.x2 = []
        self.x3 = []
        self.x4 = []

        # Start monitor thread
        if self.t_monitor:
            threading.Timer(self.THREAD_PERIOD, self.monitor_frequency).start()

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
        # Update pivot for monitoring
        self.lock.acquire()
        self.pivot = True
        self.lock.release()

        # Check if action contains NaN
        if not np.isfinite(action).any():
            action = np.zeros(np.asarray(action).shape)
            print("WARN: Found NaN in action space")

        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Store increment in action
        if np.asarray(self.action).size and np.asarray(self.last_action).size:
            self.incr_action = self.action - self.last_action

        self.last_action = self.action
        self.action = action

        # the motors only can use a 50% of the power
        limitation = False
        if limitation :
            coef_red = 3
            action /= coef_red
            action -= 1-(1/coef_red)

            action_bias = 1
            action += action_bias

        self.last_action = action
        # print(action)

        end_sim = self.sim_time

        while (end_sim - self.start_sim) <= (1 / self.FREQUENCY):
            self.obs = self.step_sim(action)
            end_sim = self.sim_time

        # print(1 / (end_sim - self.start_sim))
        self.start_sim = self.sim_time

        # Check if the observation has NaN
        if not np.isfinite(self.obs.angular_velocity_rpy).any():
            self.obs.angular_velocity_rpy = np.zeros(np.asarray(self.obs.angular_velocity_rpy).shape)
            print("WARN: Found NaN in obs.angular_velocity_rpy space")

        if not np.isfinite(self.obs.euler).any():
            self.obs.euler = np.zeros(np.asarray(self.obs.euler).shape)
            print("WARN: Found NaN in obs.euler space")

        if not np.isfinite(self.obs.motor_velocity).any():
            self.obs.motor_velocity = np.zeros(np.asarray(self.obs.motor_velocity).shape)
            print("WARN: Found NaN in obs.motor_velocity space")

        # quat = self.obs.orientation_quat
        self.speeds = self.obs.angular_velocity_rpy / (self.omega_bounds[1])
        # print(self.omega_bounds)
        # state = np.append(quat, self.speeds)

        euler_normalized = self.obs.euler / np.asarray([pi,pi,pi])
        # euler_normalized[0] += 0.;
        self.error_RPY = (self.random_euler - self.obs.euler)/(2*pi)

        # state = np.append(self.error_RPY, self.speeds)

        state = np.append(euler_normalized, self.speeds)

        # fd = open('/home/miguel/Desktop/simData.csv', 'a')
        # fd.write(str(state[0])+' '+str(state[1])+' '+str(state[2]) +' '+ str(state[3]) +' '+ str(state[4]) +' '+ str(state[5])+' ' + str(action[0]) +' '+ str(action[1]) +' '+ str(action[2]) +' '+ str(action[3]) + '\n')
        # fd.close()

        # print(state)
        # state[1] = state[1] - 0.26
        # state[2] = state[2] - 1
        # state=np.append(state_aux,self.random_quaternion)

        self.observation_history.append(np.concatenate([state, self.obs.motor_velocity]))

        [reward, done] = self.compute_reward()

        # Check if reward is NaN
        if not np.isfinite(reward):
            reward = 0.0
            print("WARN: Found NaN in reward")

        # self.animate()

        if self.sim_time >= self.max_sim_time:
            done = True
            self.last_theta_norm=0

        info = {"sim_time": self.sim_time, "sp": self.omega_target, "current_rpy": self.omega_actual}
        # print("state: ",state);

        return state, reward, done, info

    def state(self):
        """ Get the current state """

        return np.zeros(6)

    def reset(self, reset=True):
        # self.get_random_quat()
        self.observation_history = []
        self.start_sim = 0
        return super(CaRL_env, self).reset(reset=reset)

    def monitor_frequency(self):
        # Set event loop for this thread
        asyncio.set_event_loop(self.loop)

        # Main loop
        self.lock.acquire()
        if self.pivot:
            self.th_counter = 0
        else:
            self.th_counter = self.th_counter + 1
        self.lock.release()

        if self.th_counter >= 15 and not self.paused:
            # print("THREAD: Simulation has to be paused")
            self.reset(reset=-1)
            self.paused = True

        if self.th_counter < 15 and self.paused:
            self.paused = False

        self.lock.acquire()
        self.pivot = False
        self.lock.release()

        if self.t_monitor:
            threading.Timer(self.THREAD_PERIOD, self.monitor_frequency).start()

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


