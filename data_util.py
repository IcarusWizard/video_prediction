import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def save_gif(filename, video, fps=10):
    """
        save the input video to gif
        filename: String
        video: ndarray with shape [T, H, W, C]
    """
    frame = video.shape[0]
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    def writer(i):
        return ax.imshow(video[i])
    ani = animation.FuncAnimation(fig, writer, frames=frame)
    ani.save(filename, writer='imagemagick', fps=fps)
    return fig

def torch_save_gif(filename, video, fps=10):
    video = video.numpy()
    video = np.transpose(video, (0, 2, 3, 1))
    save_gif(filename, video, fps=fps)