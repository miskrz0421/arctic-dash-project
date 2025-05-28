from gymnasium.envs.registration import register

register(
    id='ArcticDashEnv',
    entry_point='gymnasium_env.envs:ArcticDashEnv',
)
