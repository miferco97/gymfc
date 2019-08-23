from gym.envs.registration import register

kwargs = {
    "memory_size": 1,
    "max_sim_time": 30
}
id = 'CaRL_GymFC-MotorVel_M4_Ep-v0'
register(
    id=id,
    entry_point='gymfc.envs:CaRL_env',
    kwargs=kwargs)
