# simulate.py
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from preisach import PreisachModel

# ------------------------------------------------------------------
# 1. USER-DEFINED INPUT SIGNAL
# ------------------------------------------------------------------
T_END = 4 * np.pi
NT    = 800
time  = np.linspace(0, T_END, NT)

def my_input(t):
    return 1.0 * (0.7 * np.sin(t) + 0.3 * np.sin(3 * t))

u_t = my_input(time)

# ------------------------------------------------------------------
# 2. PREISACH MODEL
# ------------------------------------------------------------------
model = PreisachModel(u_range=(-1.0, 1.0), n_ab=80, gamma=1.0)
f_t = np.array([model(u) for u in u_t])

# ------------------------------------------------------------------
# 3. ANIMATION
# ------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
fig.canvas.manager.set_window_title('Preisach animation – user input')

ax_u   = ax[0,0]
ax_f   = ax[0,1]
ax_lo  = ax[1,0]
ax_tri = ax[1,1]

line_u,   = ax_u.plot([], [], lw=2, color='C0')
line_f,   = ax_f.plot([], [], lw=2, color='C1')
line_lo,  = ax_lo.plot([], [], lw=2, color='C2')
tri_img   = ax_tri.imshow(
    np.zeros((model.n_ab, model.n_ab)),
    extent=[model.u_min, model.u_max, model.u_min, model.u_max],
    origin='lower', cmap='RdBu', vmin=-1, vmax=1
)

ax_u.set_xlim(0, T_END)
ax_u.set_ylim(np.min(u_t)*1.1, np.max(u_t)*1.1)
ax_u.set_xlabel('time'); ax_u.set_ylabel('u(t)')

ax_f.set_xlim(0, T_END)
ax_f.set_ylim(np.min(f_t)*1.1, np.max(f_t)*1.1)
ax_f.set_xlabel('time'); ax_f.set_ylabel('f(t)')

ax_lo.set_xlim(np.min(u_t)*1.1, np.max(u_t)*1.1)
ax_lo.set_ylim(np.min(f_t)*1.1, np.max(f_t)*1.1)
ax_lo.set_xlabel('u'); ax_lo.set_ylabel('f')

ax_tri.set_xlabel('β'); ax_tri.set_ylabel('α')
ax_tri.set_title('Preisach triangle')

def animate(n):
    line_u.set_data(time[:n+1], u_t[:n+1])
    line_f.set_data(time[:n+1], f_t[:n+1])
    line_lo.set_data(u_t[:n+1], f_t[:n+1])

    mask, extent = model.triangle_mask()
    tri_img.set_data(mask)
    tri_img.set_extent(extent)
    return line_u, line_f, line_lo, tri_img

ani = animation.FuncAnimation(fig, animate, frames=NT, interval=20, blit=True)
plt.tight_layout()
plt.show()

# Optional: save animation
# ani.save('Plots/preisach_animation.mp4', writer='ffmpeg', fps=30, dpi=150)