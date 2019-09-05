import numpy as np

class POSITION_PID():
    def __init__(self, env, kp_R, ki_R, kd_R, kp_P, ki_P, kd_P, kp_Y, ki_Y, kd_Y):
        # Env instance
        self.env = env

        # Roll PID constants
        self.kp_R = kp_R
        self.ki_R = ki_R
        self.kd_R = kd_R

        # Pitch PID constants
        self.kp_P = kp_P
        self.ki_P = ki_P
        self.kd_P = kd_P

        # Yaw PID constants
        self.kp_Y = kp_Y
        self.ki_Y = ki_Y
        self.kd_Y = kd_Y

        # Accum errors
        self.accum_error_R = 0.0
        self.accum_error_P = 0.0
        self.accum_error_Y = 0.0

        # Last error
        self.last_error_R = 0.0
        self.last_error_P = 0.0
        self.last_error_Y = 0.0

    def predict(self, obs, state_ref=np.asarray([0.0, 0.0, 0.0])):
        # Initialize PID_actions as a vector
        PID_actions = np.zeros(4)

        # Get RPY
        self.state = np.asarray(obs[0][0:3])

        # Compute errors
        error_R = self.state[0] - state_ref[0]
        error_P = self.state[1] - state_ref[1]
        error_Y = self.state[2] - state_ref[2]

        # Accum errors
        self.accum_error_R += error_R
        self.accum_error_P += error_P
        self.accum_error_Y += error_Y

        # Compute output
        P_error = [self.kp_R * error_R, self.kp_P * error_P, self.kp_Y * error_Y]
        I_error = [self.ki_R * self.accum_error_R, self.ki_P * self.accum_error_P, self.ki_Y * self.accum_error_Y]
        D_error = [self.kd_R * (error_R - self.last_error_R), self.kd_P * (error_P - self.last_error_P), self.kd_Y * (error_Y - self.last_error_Y)]

        # Store current error
        self.last_error_R = error_R
        self.last_error_P = error_P
        self.last_error_Y = error_Y

        PID_outputs = P_error + I_error + D_error

        # PID_outputs to motor outputs conversion (Motor Mix)
        PID_actions[0] = + PID_outputs[1] + PID_outputs[0] + PID_outputs[2]
        PID_actions[1] = - PID_outputs[1] + PID_outputs[0] - PID_outputs[2]
        PID_actions[2] = + PID_outputs[1] - PID_outputs[0] - PID_outputs[2]
        PID_actions[3] = - PID_outputs[1] - PID_outputs[0] + PID_outputs[2]

        # Action Clipping
        PID_actions = np.clip(PID_actions, -1, 1)

        return np.asarray([PID_actions]), self.state

