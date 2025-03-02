import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import numpy as np

from dm_control import manipulation

import mjmanip.envs.mylift

def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(
        fig=fig, func=update, frames=frames, interval=interval, blit=True, repeat=False
    )
    return HTML(anim.to_html5_video())


env = manipulation.load("my_lift_vision", seed=42)
action_spec = env.action_spec()


def sample_random_action():
    return env.random_state.uniform(
        low=action_spec.minimum,
        high=action_spec.maximum,
    ).astype(action_spec.dtype, copy=False)


# Step the environment through a full episode using random actions and record
# the camera observations.
frames = []
timestep = env.reset()
frames.append(timestep.observation["front_flat_far"])
while not timestep.last():
    timestep = env.step(sample_random_action())
    frames.append(timestep.observation["front_flat_far"])
all_frames = np.concatenate(frames, axis=0)
display_video(all_frames, 30)
