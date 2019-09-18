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
        self.ACTION_SMOOTHING = False
        self.BUFFER_SIZE = 5
        self.action_buffer = np.zeros((4, self.BUFFER_SIZE))

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

    def compute_reward_type_a(self):
        # Basic reward of angles

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

    def compute_reward_type_b(self):
        # Basic reward of angles and increment in actions

        # Define rewards
        rew_incr_action = 0
        rew_R = np.abs((self.obs.euler[0])/(pi))
        rew_P = np.abs((self.obs.euler[1])/(pi))
        rew_Y = np.abs((self.obs.euler[2])/(pi))

        if np.asarray(self.incr_action).size:
            rew_incr_action = 0.1 * - np.sqrt(np.power(np.clip(np.sum(np.abs(self.incr_action)) / np.asarray(self.incr_action).size, a_min=0, a_max=1), 2))

        if np.abs(self.obs.euler[0]) >= 0.999 * (pi / 2) or np.abs(self.obs.euler[1]) >= 0.999 * (pi / 2):
            done = True
            reward = -1
        else:
            rew_full_angle = np.power(1 - np.clip(((rew_P + rew_R + rew_Y) / 3),0,1),3)
            reward = rew_full_angle + rew_incr_action
            # print("Reward action: " + str(rew_incr_action))
            # print("Reward angle: " + str(rew_full_angle))
            done = False

        return [reward, done]

    def compute_reward(self):
        if not self.REAL_FLIGHT:
            # Compute reward
            # [reward, done] = self.compute_reward_type_a()
            [reward, done] = self.compute_reward_type_b()

            # Store last reward
            if done:
                self.last_reward = 0
                self.last_action = []
                self.action = []

            return [reward, done]
        else:
            return [0, False]

    def compute_state_type_a(self, state_zero=False):
        # State composed of absolute angles and absolute velocities
        if not state_zero:
            # quat = self.obs.orientation_quat
            self.speeds = self.obs.angular_velocity_rpy / (self.omega_bounds[1])

            euler_normalized = self.obs.euler / np.asarray([pi, -pi, pi])
            # euler_normalized[0] += 0.;
            self.error_RPY = (self.random_euler - self.obs.euler)/(2*pi)

            # state = np.append(self.error_RPY, self.speeds)

            state = np.append(euler_normalized, self.speeds)
        else:
            state = np.zeros(6)

        return state

    def compute_state_type_b(self, state_zero=False):
        if not state_zero:
            # State composed of absolute angles, absolute velocities and last action increment
            self.speeds = self.obs.angular_velocity_rpy / (self.omega_bounds[1])

            euler_normalized = self.obs.euler / np.asarray([pi, -pi, pi])

            state = np.append(euler_normalized, self.speeds)

            # Include last action
            if np.asarray(self.incr_action).size:
                state = np.append(state, self.incr_action)
            else:
                state = np.append(state, np.zeros(4))
            # print("State: " + str(state))
        else:
            state = np.zeros(10)

        return state

    def compute_state_type_real(self, state_zero=False):
        if not state_zero:
            # State composed of absolute angles, absolute velocities and last action increment
            self.speeds = self.obs.angular_velocity_rpy

            euler_normalized = self.obs.euler

            state = np.append(euler_normalized, self.speeds)

        else:
            state = np.zeros(6)

        return state

    def compute_state(self, state_zero=False):
        if self.REAL_FLIGHT:
            state = self.compute_state_type_real(state_zero=state_zero)
        else:
            state = self.compute_state_type_a(state_zero=state_zero)
            # state = self.compute_state_type_b(state_zero=state_zero)


        return state

    def filter_action(self, action):
        self.action_buffer = np.roll(self.action_buffer, 1, axis=1)
        self.action_buffer[:, 0] = action

        return np.ma.average(self.action_buffer, axis=1)



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
        self.FREQUENCY = float(kwargs["frequency"])
        print("Agent frequency: " + str(self.FREQUENCY))
        self.THREAD_PERIOD = (1.0 / (10.0 * self.FREQUENCY)) # 10 times faster
        self.th_counter = 0
        self.pivot = False
        self.paused = False
        self.lock = threading.RLock()
        self.start_sim = 0

        self.past_error=np.zeros(3)

        self.obs = self.step_sim(np.zeros(self.action_space.shape))


        # Start monitor thread
        if self.t_monitor:
            threading.Timer(self.THREAD_PERIOD, self.monitor_frequency).start()

    def step(self, action):
        # Update pivot for monitoring
        self.lock.acquire()
        self.pivot = True
        self.lock.release()

        # Store local variable
        action_local = np.asarray(action)

        # Check if action contains NaN
        if not np.isfinite(action_local).any():
            action_local = np.zeros(np.asarray(action_local).shape)
            print("WARN: Found NaN in action space")

        # Clip actions
        action_local = np.clip(action_local, self.action_space.low, self.action_space.high)

        # Filter action
        if self.ACTION_SMOOTHING:
            action_local = self.filter_action(action_local)

        # Store increment in action
        if np.asarray(self.action).size and np.asarray(self.last_action).size:
            self.incr_action = (self.action - self.last_action) / 2

        self.last_action = self.action
        self.action = action_local


        end_sim = self.sim_time

        while (end_sim - self.start_sim) <= (1 / self.FREQUENCY):
            # print(action_local)
            self.obs = self.step_sim(action_local, send_actions=False)
            end_sim = self.sim_time
            # print(self.start_sim)
            # print(end_sim)

        self.obs = self.step_sim(action_local)

        # print("Measured frequency: " + str(1 / (end_sim - self.start_sim)))
        # print("Measured end: " + str(end_sim) + "\t Masured start: " + str(self.start_sim))
        self.start_sim = self.sim_time


        try:

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

            state = self.compute_state()

            self.observation_history.append(np.concatenate([state, self.obs.motor_velocity]))

            [reward, done] = self.compute_reward()

            # Check if reward is NaN
            if not np.isfinite(reward):
                reward = 0.0
                print("WARN: Found NaN in reward")

            # self.animate()

            if not self.REAL_FLIGHT:
                if self.sim_time >= self.max_sim_time:
                    done = True
                    self.last_theta_norm = 0

        except:
            info = {"forwarded_action": action_local, "sim_time": self.sim_time, "sp": self.omega_target,
                    "current_rpy": self.omega_actual}
            # print("state: ",state);



            return state, 0, False, info

        info = {"forwarded_action": action_local, "sim_time": self.sim_time, "sp": self.omega_target, "current_rpy": self.omega_actual}
        # print("state: ",state);

        return state, reward, done, info

    def state(self):
        """ Get the current state """

        return self.compute_state(state_zero=True)

    def reset(self, reset=True):
        # self.get_random_quat()
        self.observation_history = []
        self.start_sim = 0
        self.last_action = []
        self.last_reward = 0
        self.incr_action = []
        self.action_buffer = np.zeros((4, self.BUFFER_SIZE))

        self.accum_error=np.zeros(3)
        self.past_error = np.zeros(3)

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