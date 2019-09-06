import numpy as np

class POS_VEL_PID():
    def __init__(self, env, kp_R_pos, kp_P_pos, kp_Y_pos, kp_R_vel, ki_R_vel, kd_R_vel, kp_P_vel, ki_P_vel, kd_P_vel, kp_Y_vel, ki_Y_vel, kd_Y_vel, f=70.0):
        # Env instance
        self.env = env

        # Roll PID constants
        self.kp_R_pos = - kp_R_pos

        # Pitch PID constants
        self.kp_P_pos = kp_P_pos

        # Yaw PID constants
        self.kp_Y_pos = kp_Y_pos

        # Accum errors
        self.accum_error_R_pos = 0.0
        self.accum_error_P_pos = 0.0
        self.accum_error_Y_pos = 0.0

        # Last error
        self.last_error_R_pos = 0.0
        self.last_error_P_pos = 0.0
        self.last_error_Y_pos = 0.0

        # Roll PID constants
        self.kp_R_vel = kp_R_vel
        self.ki_R_vel = ki_R_vel
        self.kd_R_vel = kd_R_vel

        # Pitch PID constants
        self.kp_P_vel = - kp_P_vel
        self.ki_P_vel = ki_P_vel
        self.kd_P_vel = kd_P_vel

        # Yaw PID constants
        self.kp_Y_vel = - kp_Y_vel
        self.ki_Y_vel = ki_Y_vel
        self.kd_Y_vel = kd_Y_vel

        # Accum errors
        self.accum_error_R_vel = 0.0
        self.accum_error_P_vel = 0.0
        self.accum_error_Y_vel = 0.0

        # Last error
        self.last_error_R_vel = 0.0
        self.last_error_P_vel = 0.0
        self.last_error_Y_vel = 0.0

        # Frequency
        self.f = f

    def predict(self, obs, state_ref=np.asarray([0.0, 0.0, 0.0])):
        # Initialize PID_actions as a vector
        PID_actions = np.zeros(4)

        # Get RPY
        self.state_pos = np.asarray(obs[0][0:3])

        # Compute errors
        error_R_pos = self.state_pos[0] - state_ref[0]
        error_P_pos = self.state_pos[1] - state_ref[1]
        error_Y_pos = self.state_pos[2] - state_ref[2]

        # Accum errors
        self.accum_error_R_pos += error_R_pos
        self.accum_error_P_pos += error_P_pos
        self.accum_error_Y_pos += error_Y_pos

        # Compute output
        P_error_pos = np.asarray([self.kp_R_pos * error_R_pos, self.kp_P_pos * error_P_pos, self.kp_Y_pos * error_Y_pos])

        # Store current error
        self.last_error_R_pos = error_R_pos
        self.last_error_P_pos = error_P_pos
        self.last_error_Y_pos = error_Y_pos

        PID_outputs_pos = P_error_pos

        # Get RPY
        self.state_vel = np.asarray(obs[0][3:6])

        # Compute errors
        error_R_vel = self.state_vel[0] - PID_outputs_pos[0]
        error_P_vel = self.state_vel[1] - PID_outputs_pos[1]
        error_Y_vel = self.state_vel[2] - PID_outputs_pos[2]

        # Accum errors
        self.accum_error_R_vel += error_R_vel
        self.accum_error_P_vel += error_P_vel
        self.accum_error_Y_vel += error_Y_vel

        # Compute output
        P_error_vel = np.asarray([self.kp_R_vel * error_R_vel, self.kp_P_vel * error_P_vel, self.kp_Y_vel * error_Y_vel])
        I_error_vel = np.asarray([self.ki_R_vel * self.accum_error_R_vel, self.ki_P_vel * self.accum_error_P_vel, self.ki_Y_vel * self.accum_error_Y_vel])
        D_error_vel = np.asarray([self.kd_R_vel * (error_R_vel - self.last_error_R_vel) * self.f, self.kd_P_vel * (error_P_vel - self.last_error_P_vel) * self.f, self.kd_Y_vel * (error_Y_vel - self.last_error_Y_vel) * self.f])

        # Store current error
        self.last_error_R_vel = error_R_vel
        self.last_error_P_vel = error_P_vel
        self.last_error_Y_vel = error_Y_vel

        PID_outputs = P_error_vel + I_error_vel + D_error_vel

        # PID_outputs to motor outputs conversion (Motor Mix)
        PID_actions[0] = + PID_outputs[1] + PID_outputs[0] + PID_outputs[2]
        PID_actions[1] = - PID_outputs[1] + PID_outputs[0] - PID_outputs[2]
        PID_actions[2] = + PID_outputs[1] - PID_outputs[0] - PID_outputs[2]
        PID_actions[3] = - PID_outputs[1] - PID_outputs[0] + PID_outputs[2]

        # Action Clipping
        PID_actions = np.clip(PID_actions, -1, 1)

        return np.asarray([PID_actions]), np.append(self.state_pos, self.state_vel)


class POSITION_PID():
    def __init__(self, env, kp_R, ki_R, kd_R, kp_P, ki_P, kd_P, kp_Y, ki_Y, kd_Y, f=70.0):
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
        self.kd_Y = - kd_Y

        # Accum errors
        self.accum_error_R = 0.0
        self.accum_error_P = 0.0
        self.accum_error_Y = 0.0

        # Last error
        self.last_error_R = 0.0
        self.last_error_P = 0.0
        self.last_error_Y = 0.0

        # Frequency
        self.f = f

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
        P_error = np.asarray([self.kp_R * error_R, self.kp_P * error_P, self.kp_Y * error_Y])
        I_error = np.asarray([self.ki_R * self.accum_error_R, self.ki_P * self.accum_error_P, self.ki_Y * self.accum_error_Y])
        D_error = np.asarray([self.kd_R * (error_R - self.last_error_R) * self.f, self.kd_P * (error_P - self.last_error_P) * self.f, self.kd_Y * (error_Y - self.last_error_Y) * self.f])

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

