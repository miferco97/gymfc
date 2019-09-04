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
        self.PID_activated = True
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
        # Compute reward
        # [reward, done] = self.compute_reward_type_a()
        [reward, done] = self.compute_reward_type_b()

        # Store last reward
        if done:
            self.last_reward = 0
            self.last_action = []
            self.action = []

        return [reward, done]

    def compute_state_type_a(self, state_zero=False):
        # State composed of absolute angles and absolute velocities
        if not state_zero:
            # quat = self.obs.orientation_quat
            self.speeds = self.obs.angular_velocity_rpy / (self.omega_bounds[1])

            euler_normalized = self.obs.euler / np.asarray([pi,pi,pi])
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

            euler_normalized = self.obs.euler / np.asarray([pi,pi,pi])

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

    def compute_state(self, state_zero=False):
        # state = self.compute_state_type_a(state_zero=state_zero)
        state = self.compute_state_type_b(state_zero=state_zero)

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
        self.FREQUENCY = 70.0
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

        # Check if action contains NaN
        if not np.isfinite(action).any():
            action = np.zeros(np.asarray(action).shape)
            print("WARN: Found NaN in action space")

        # Clip actions
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Filter action
        if self.ACTION_SMOOTHING:
            action = self.filter_action(action)

        # Store increment in action
        if np.asarray(self.action).size and np.asarray(self.last_action).size:
            self.incr_action = (self.action - self.last_action) / 2

        self.last_action = self.action
        self.action = action

        end_sim = self.sim_time

        while (end_sim - self.start_sim) <= (1 / self.FREQUENCY):
            # If PID is active
            if self.PID_activated:
                action = self.PID_actions()

            # print(action)
            self.obs = self.step_sim(action, send_actions=False)
            end_sim = self.sim_time

        # print(action)
        self.obs = self.step_sim(action)

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

            if self.sim_time >= self.max_sim_time:
                done = True
                self.last_theta_norm = 0

        except:
            info = {"forwarded_action": action, "sim_time": self.sim_time, "sp": self.omega_target,
                    "current_rpy": self.omega_actual}
            # print("state: ",state);

            return state, 0, False, info

        info = {"forwarded_action": action, "sim_time": self.sim_time, "sp": self.omega_target, "current_rpy": self.omega_actual}
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

    def PID_actions(self):

        #if self.PID_activated is TRUE


        # vectors Index Roll=0, Pitch=1,Yaw=0

        # ***** PID simulation gains ******
        #PID constants
        # Kp = np.asarray([0.9,0.9,0.5])
        # Ki = np.asarray([0.0001,0.0001,0.0001])
        # Kd = np.asarray([0.0,0.0,0.0])

        # ***** PID real-flight gains ******
        #PID constants
        Kp = np.asarray([6.0,6.0,0.5])
        # Ki = np.asarray([0.0001,0.0001,0.0001])
        Ki = np.asarray([0.0,0.0,0.0])
        Kd = np.asarray([2.6,2.6,0.0])

        #initialize PID_actions as a vector
        PID_actions = np.zeros(4)

        #normalize state (- sign in pitch for coherence in errors)
        state = self.obs.euler/np.asarray([pi,-pi,pi])

        #reference state for the PID
        state_ref=np.asarray([0.0,0.0,0.0])

        #vectorized state computing
        error = state-state_ref

        #Proportional parts
        P_error = Kp*error

        #accumulated error for integral
        self.accum_error += error

        # Integral part

        I_error= Ki * self.accum_error

        D_error = Kd * (error - self.past_error)
        self.past_error = error

        #PI Contribution
        # D isn't implemented yet

        PID_outputs = P_error + I_error + D_error

        #PID_outputs to motor outputs conversion (Motor Mix)

        PID_actions[0] = + PID_outputs[1] + PID_outputs[0] + PID_outputs[2]
        PID_actions[1] = - PID_outputs[1] + PID_outputs[0] - PID_outputs[2]
        PID_actions[2] = + PID_outputs[1] - PID_outputs[0] - PID_outputs[2]
        PID_actions[3] = - PID_outputs[1] -PID_outputs[0]  + PID_outputs[2]

        #action Clipping

        PID_actions=np.clip(PID_actions,-1,1)

        return PID_actions

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


