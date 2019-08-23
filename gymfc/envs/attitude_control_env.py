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

class CaRL_env(GazeboEnv):
    def __init__(self, **kwargs):

        self.index = 0
        self.max_sim_time = kwargs["max_sim_time"]
        self.BUFFER_SIZE = 5
        np.random.seed(seed=None)

        self.observation_history = []
        self.memory_size = kwargs["memory_size"]
        super().__init__()


        self.action=[]
        self.omega_target = []
        self.last_action = np.asarray([0.0,0.0,0.0,0.0])
        self.start = 0
        self.th_counter = 0
        self.pivot = False
        self.paused = False

        self.lock = threading.RLock()
        self.start_sim = 0
        self.time_t=0

        self.obs = self.step_sim(np.zeros(self.action_space.shape))
        self.render()

        # Flags for changes in the enviroment.
        self.discountFactor = 0.9
        self.ACTION_SMOOTHING = True
        self.PID_activated = False
        self.FREQUENCY = 70.0
        self.THREAD_PERIOD = (1.0 / (10.0 * self.FREQUENCY)) # 10 times faster

        # Start monitor thread
        if self.t_monitor:
            threading.Timer(self.THREAD_PERIOD, self.monitor_frequency).start()


    def compute_state(self, state_zero=False):
        # State composed of absolute angles and absolute velocities
        if not state_zero:
            #change sign of pitch for error coherence
            self.speeds = self.obs.angular_velocity_rpy / np.asarray([2*pi,-2*pi,2*pi]) # self.omega_bounds => 2*pi =6.283..
            euler_normalized = self.obs.euler / np.asarray([pi,-pi,pi])
            state = np.append(euler_normalized, self.speeds)

        else:
            state = np.zeros(6)

        return state


    def compute_reward(self, state):
        # Compute reward
<<<<<<< HEAD
        rew_R = np.abs(state[0]) + 0.1 * np.abs(state[3])
        rew_P = 0# state[1]
        rew_Y = 0#state[2]
=======
        rew_R = np.abs(state[0])
        rew_P = 0 #state[1]
        rew_Y = 0 #state[2]
>>>>>>> 06a3abce444dec006c7fe935d286ed16895a3a96

        if np.abs(self.obs.euler[0]) >= 0.999 * (pi / 2) or np.abs(self.obs.euler[1]) >= 0.999 * (pi / 2):
            done = True
            reward = -1
        else:
            done = False
            reward = np.power(1 - np.clip(((rew_P + rew_R + rew_Y) / 3), 0, 1), 2)

        return [reward, done]

    def filter_action(self, action):
        self.action_buffer = np.roll(self.action_buffer, 1, axis=1)
        self.action_buffer[:, 0] = action

        return np.ma.average(self.action_buffer, axis=1)

    def step(self, action):
        # Update pivot for monitoring
        self.lock.acquire()
        self.pivot = True
        self.lock.release()

        # Check if action contains NaN

        if self.PID_activated:
            action = self.PID_actions(self.state)

        if not np.isfinite(action).any():
            action = np.zeros(np.asarray(action).shape)
            print("WARN: Found NaN in action space")

        # Clip actions
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Filter action
        if self.ACTION_SMOOTHING:
            action = self.filter_action(action)

        # Store increment in action

        # if np.asarray(self.action).size and np.asarray(self.last_action).size:
        #     self.incr_action = (self.action - self.last_action) / 2

        self.last_action = self.action
        self.action = action

        end_sim = self.sim_time

        # Enables
        while (end_sim - self.start_sim) <= (1 / self.FREQUENCY):
            self.obs = self.step_sim(action)
            end_sim = self.sim_time

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

        self.state = self.compute_state()

        self.observation_history.append(np.concatenate([self.state, self.obs.motor_velocity]))

        [reward, done] = self.compute_reward(self.state)

        # Check if reward is NaN
        if not np.isfinite(reward):
            reward = 0.0
            print("WARN: Found NaN in reward")

        # self.animate()

        if self.sim_time >= self.max_sim_time:
            done = True
            # self.last_theta_norm=0

        info = {"forwarded_action": action, "sim_time": self.sim_time, "sp": self.omega_target, "current_rpy": self.omega_actual}
        # print("state: ",state);

        return self.state, reward, done, info

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

        self.past_error=np.zeros(3)
        self.accum_error = np.zeros(3)

        return super().reset(reset=reset)

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
            self.reset(reset=False)
            self.paused = True

        if self.th_counter < 15 and self.paused:
            self.paused = False

        self.lock.acquire()
        self.pivot = False
        self.lock.release()

        if self.t_monitor:
            threading.Timer(self.THREAD_PERIOD, self.monitor_frequency).start()

    def PID_actions(self,state):

        #if self.PID_activated is TRUE
        # vectors Index Roll=0, Pitch=1,Yaw=0

        #PID constants
        Kp = np.asarray([100.0,0.0,0.0])
        Ki = np.asarray([0.0,0.000,0.000])
        Kd = np.asarray([10.0,0.0,0.0])

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
        I_error=Ki*self.accum_error;

        #PI Contribution
        # D isn't implemented yet
        D_error=Kd*(error-self.past_error)/(time.time()-self.time_t+1e-12)
        self.time_t = time.time()

        self.past_error=error;
        PID_outputs = P_error + I_error + D_error

        #PID_outputs to motor outputs conversion (Motor Mix)

        PID_actions[0] = + PID_outputs[1] + PID_outputs[0] + PID_outputs[2]
        PID_actions[1] = - PID_outputs[1] + PID_outputs[0] - PID_outputs[2]
        PID_actions[2] = + PID_outputs[1] - PID_outputs[0] - PID_outputs[2]
        PID_actions[3] = - PID_outputs[1] - PID_outputs[0] + PID_outputs[2]

        #action Clipping

        PID_actions=np.clip(PID_actions,-1,1)
        print(PID_actions)
        return PID_actions
