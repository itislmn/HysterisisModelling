# simulate.py
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from preisach import PreisachModel

# Input signal
T_END = 4 * np.pi
NT = 800
time = np.linspace(0, T_END, NT)

def my_input(t):
    return 0.7 * np.sin(t) + 0.3 * np.sin(3 * t)

u_t = my_input(time)

# Model
model = PreisachModel(u_range=(-1.0, 1.0), gamma=1.0)
f_t = np.array([model(u) for u in u_t])

# Animation
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
fig.canvas.manager.set_window_title('Preisach Model – TU Darmstadt')

(line_u,) = ax[0,0].plot([], [], 'b-', lw=2)
(line_f,) = ax[0,1].plot([], [], 'r-', lw=2)
(line_lo,) = ax[1,0].plot([], [], 'g-', lw=2)
(tri,) = ax[1,1].plot([], [], 'k.')

ax[0,0].set_xlim(0, T_END); ax[0,0].set_ylim(-1.1, 1.1); ax[0,0].set_ylabel('u(t)')
ax[0,1].set_xlim(0, T_END); ax[0,1].set_ylim(-1.1, 1.1); ax[0,1].set_ylabel('f(t)')
ax[1,0].set_xlim(-1.1, 1.1); ax[1,0].set_ylim(-1.1, 1.1); ax[1,0].set_xlabel('u'); ax[1,0].set_ylabel('f')
ax[1,1].set_visible(False)  # disable triangle for now

def animate(i):
    line_u.set_data(time[:i+1], u_t[:i+1])
    line_f.set_data(time[:i+1], f_t[:i+1])
    line_lo.set_data(u_t[:i+1], f_t[:i+1])
    return line_u, line_f, line_lo

ani = animation.FuncAnimation(fig, animate, frames=NT, interval=20, blit=True)
plt.tight_layout()
plt.show()

ani.save('Plots/preisach_animation.mp4', writer='ffmpeg', fps = 30, dpi = 150)