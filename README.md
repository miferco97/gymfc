# GymFC

GymFC is an [OpenAI Gym](https://github.com/openai/gym) environment specifically
designed for
developing intelligent flight control systems using reinforcement learning. This
environment is meant to serve as a tool for researchers to benchmark their
controllers to progress the state-of-the art of intelligent flight control.
Our tech report is available at [https://arxiv.org/abs/1804.04154](https://arxiv.org/abs/1804.04154)  providing details of the
environment and  benchmarking of PPO, TRPO and DDPG using [OpenAI Baselines](https://github.com/openai/baselines). We compare the performance results to a PID controller and find PPO to out perform PID in regards to rise time and overall error. Please use the following BibTex entry to cite our
work,
```
@misc{1804.04154,
Author = {William Koch and Renato Mancuso and Richard West and Azer Bestavros},
Title = {Reinforcement Learning for UAV Attitude Control},
Year = {2018},
Eprint = {arXiv:1804.04154},
}
```

# Installation 
Note, Ubuntu is the only OS currently supported. Please submit a PR for the
README.md if you are
able to get it working on another platform.   
1. Download and install [Gazebo 9](http://gazebosim.org/download). Tested on Ubuntu 16.04 LTS and 18.04 LTS.
2. From root directory of this project, `pip3 install .`

# Iris PID Example
To verify you have installed the environment correctly it is recommended to run
the supplied PID controller controlling an Iris quadcopter model. This example
uses the configuration file `examples/config/iris.config`. Before running the
example verify the configuration, specifically that the Gazebo `SetupFile` is pointing to the correct location.
The example requires additional Python modules, from `examples` run,
```
pip3 install -r requirements.txt
```
To run the example change directories to `examples/controllers` and execute,
```
python3 run_iris_pid.py
```
If your environment is installed properly you should observe a plot that
closely resembles this step response,
![PID Step
Response](https://raw.githubusercontent.com/wil3/gymfc/master/images/pid-step-AttFC_GyroErr-MotorVel_M4_Ep-v0.png)


# Development 

GymFC's primary goal is to train controllers capable of flight in the real-world. 
 In order to construct optimal flight controllers, the aircraft used in
simulation should closely match the real-world aircraft. Therefore the GymFC environment is decoupled from the simulated aircraft.
As previously mentioned, GymFC comes with an example to verify the environment.
The Iris model can be useful for testing out new controllers. However when
transferring the controller to run on a different aircraft, a new model will be
required. Once the model is developed set the model directory to `AircraftModel` in your configuration file.

It is recommended to run GymFC in headless mode (i.e. using `gzserver`) however
during development and testing it may be desired to visually see the aircraft.  You can do this by using the `render` OpenAI gym API call which will also start `gzclient` along side `gzserver`. For example when creating the environment use,
```
env = gym.make(env_id)
env.render()
```
[![GymFC Visualization Demo](https://raw.githubusercontent.com/wil3/gymfc/master/images/gymfc-vis.png)](https://youtu.be/sX9NwmDg6SA)

If you plan to work with the GymFC source code you will want to install it in
development mode, `pip3 install -e .` from the root directory. You will also
need to build the plugin manually by running the script
`gymfc/envs/assets/gazebo/plugins/build_plugin.sh`. 

# Training
Install Anaconda environment with python3.5 and run:
```
export PATH=/home/alejandro/anaconda2/bin:$PATH

source activate <environment_name>
```
Make necessary exports and run the training:
```
export PYTHONPATH=$PYTHONPATH:<path_to_gymfc>

export GYMFC_CONFIG=<path_to_gymfc>/examples/configs/iris.config

OPENAI_LOG_FORMAT=tensorboard,log,stdout python -m baselines.run --alg=ppo2 --env=CaRL_GymFC-MotorVel_M4_Ep-v0 --num_timesteps=1e12
```
For loading a pre-trained model, execute:

```
OPENAI_LOG_FORMAT=tensorboard,log,stdout python -m baselines.run --alg=ppo2 --env=CaRL_GymFC-MotorVel_M4_Ep-v0 --num_timesteps=1e12 --load_path=<path_to_checkpoint>
```

# Environments

Different environments are available depending on the capabilities of the flight
control system. For example new ESCs contain sensors to provide telemetry
including the velocity of the rotor which can be used as additional state in the
environment. Environment naming format is [prefix]\_[inputs]\_M[actuator
count]\_[task type] where prefix=AttFC, Ep is episodic tasks, and Con is
continuous tasks.

## AttFC_GyroErr-MotorVel_M4_Ep-v0

This environment is an episodic task to learn attitude control of a quadcopter. At the beginning of each episode the
quadcopter is at rest. A random angular velocity is sampled and the agent must achieve this target  within
1 second. 

**Observation Space** Box(7,) where 3 observations correspond to the angular velocity error for each axis in radians/second (i.e Ω\* − Ω) in range [-inf, inf] and 4 observations correspond
to the angular velocity of each rotor in range [-inf, inf].
 
**Action Space** Box(4,) corresponding to each PWM value to be sent to the ESC in
the range [-1, 1].

**Reward** The error normalized between [-1, 0] representing how close the angular velocity is
to the target calculated by -clip(sum(|Ω\* − Ω |)/3Ω\_max)  where the clip
function bounds the result to [-1, 0] and  Ω\_max is the initially error from
when the target angular velocity is set.

Note: In the referenced paper different memory sizes were tested, however for PPO it was
found additional memory did not help. At the moment for research, debugging and testing purposes environments with different memory sizes are included and can be referenced by AttFC_GyroErr1-MotorVel_M4_Ep-v0 - AttFC_GyroErr10-MotorVel_M4_Ep-v0.

## AttFC_GyroErr-MotorVel_M4_Con-v0

This environment is essentially the same as the episodic variant however it runs
for 60 seconds and continually changes the target angular velocities randomly
between [0.1, 1] seconds.

## AttFC_GyroErr1_M4_Ep-v0 - AttFC_GyroErr10_M4_Ep-v0

This environment supports ESCs without telemetry and only relies on the gyro
readings as environment observations. Preliminary testing has shown memory > 1
increases accuracy. 
