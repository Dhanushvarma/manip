from gymnasium.envs.registration import register


register(
    id="manip/WidowEnv-v0",
    entry_point="manip.envs:WidowEnv",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)
