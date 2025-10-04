#author itislmn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from scipy.interpolate import LinearNDInterpolator
from typing import Tuple, Callable, List
import copy

def analyticalPreisachFunction2(A: float, Hc: float, sigma: float, beta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    nom1 = 1
    den1 = 1 + ((beta - Hc) * sigma / Hc) ** 2
    nom2 = 1
    den2 = 1 + ((alpha + Hc) * sigma / Hc) ** 2
    preisach = A * (nom1 / den1) * (nom2 / den2)
    # Zero out lower-right triangle (α < β region)
    for i in range(preisach.shape[0]):
        preisach[i, (-i - 1):] = 0
    return preisach

def removeInBetween(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(arr) < 3:
        return arr, np.ones(len(arr), dtype=bool)
    keep = np.ones(len(arr), dtype=bool)
    for i in range(1, len(arr) - 1):
        if arr[i] == arr[i - 1] == arr[i + 1]:
            keep[i] = False
    return arr[keep], keep

def removeRedundantPoints(pointsX: np.ndarray, pointsY: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pointsX, mask = removeInBetween(pointsX)
    pointsY = pointsY[mask]
    pointsY, mask = removeInBetween(pointsY)
    pointsX = pointsX[mask]
    return pointsX, pointsY

def preisachIntegration(w: float, Z: np.ndarray) -> np.ndarray:
    flipped = np.fliplr(np.flipud(w * Z))
    flipped_integral = np.cumsum(np.cumsum(flipped, axis=0), axis=1)
    return np.fliplr(np.flipud(flipped_integral))

class PreisachModel:
    def __init__(self, n: int, alpha0: float) -> None:
        self.n = n
        self.alpha0 = alpha0
        self.beta0 = alpha0
        x = np.linspace(-self.beta0, self.beta0, n - 1)
        y = np.linspace(-self.alpha0, self.alpha0, n - 1)
        self.width = 2 * alpha0 / (n - 1)
        self.gridX, self.gridY = np.meshgrid(x, y)
        self.gridY = np.flipud(self.gridY)  # align with Preisach triangle definition

        self.interfaceX = np.array([-self.beta0, -self.beta0], dtype=np.float64)
        self.interfaceY = np.array([-self.alpha0, -self.alpha0], dtype=np.float64)
        self.historyInterfaceX: List[np.ndarray] = []
        self.historyInterfaceY: List[np.ndarray] = []
        self.historyU = np.array([-self.alpha0], dtype=np.float64)
        self.historyOut = np.array([], dtype=np.float64)
        self.state = 'ascending'
        self.stateOld = 'ascending'
        self.stateChanged = False
        self.everett = lambda beta, alpha: np.zeros_like(beta)

    def __call__(self, u: float) -> float:
        if u > self.historyU[-1]:
            self.state = 'ascending'
        elif u < self.historyU[-1]:
            self.state = 'descending'
        else:
            self.state = self.stateOld

        self.stateChanged = (self.state != self.stateOld)

        # Current interface (without current u)
        pointsX = self.interfaceX[:-1].copy()
        pointsY = self.interfaceY[:-1].copy()

        if self.stateChanged:
            pointsX = np.append(pointsX, self.historyU[-1])
            pointsY = np.append(pointsY, self.historyU[-1])

        if self.state == 'ascending':
            pointsY[pointsY <= u] = u
            pointsY[-1] = u
        elif self.state == 'descending':
            pointsX[pointsX >= u] = u
            pointsX[-1] = u

        self.interfaceX = np.append(pointsX, u)
        self.interfaceY = np.append(pointsY, u)
        self.interfaceX, self.interfaceY = removeRedundantPoints(self.interfaceX, self.interfaceY)

        self.stateOld = self.state
        self.historyInterfaceX.append(self.interfaceX.copy())
        self.historyInterfaceY.append(self.interfaceY.copy())
        self.historyU = np.append(self.historyU, u)
        output = self.calculateOutput()
        self.historyOut = np.append(self.historyOut, output)
        return output

    def resetHistory(self) -> None:
        self.historyInterfaceX = []
        self.historyInterfaceY = []
        self.historyU = np.array([-self.alpha0], dtype=np.float64)
        self.historyOut = np.array([], dtype=np.float64)
        self.state = 'ascending'
        self.stateOld = 'ascending'
        self.stateChanged = False

    def setNegSatState(self) -> None:
        self.interfaceX = np.array([-self.beta0, -self.beta0], dtype=np.float64)
        self.interfaceY = np.array([-self.alpha0, -self.alpha0], dtype=np.float64)
        self.resetHistory()

    def setDemagState(self, n: int = 150) -> None:
        self.setNegSatState()
        excitation = np.linspace(1, 0, n)
        excitation[1::2] = -excitation[1::2]
        for val in excitation:
            self(val * self.alpha0)
        self.resetHistory()

    def invert(self) -> 'PreisachModel':
        invModel = PreisachModel(self.n, self.alpha0)
        print('Inverting Model...')

        FODs = []
        Mk = []
        mk = []
        invEverettVals = []

        alphas = np.linspace(-self.alpha0, self.alpha0, self.n - 1)
        for alpha in alphas:
            betas = np.linspace(-self.beta0, alpha, max(1, int((alpha + self.alpha0) / self.width)))
            for beta in betas:
                self.setNegSatState()
                out1 = self(alpha)
                out2 = self(beta)
                FODs.append((alpha, beta))
                Mk.append(out1)
                mk.append(out2)
                invEverettVals.append(0.5 * (alpha - beta))

        if not FODs:
            raise RuntimeError("No FODs generated for inversion.")

        points = np.column_stack((mk, Mk))
        Z = np.array(invEverettVals)
        invEverettInterp = LinearNDInterpolator(points, Z, fill_value=0.0)
        invModel.setEverettFunction(invEverettInterp)
        print('Model inversion successful!')
        return invModel

    def calculateOutput(self) -> float:
        total = 0.0
        for i in range(1, len(self.interfaceX)):
            Mk = self.interfaceY[i]
            mk = self.interfaceX[i]
            mkOld = self.interfaceX[i - 1]
            total += self.everett(mkOld, Mk) - self.everett(mk, Mk)
        return -self.everett(-self.beta0, self.alpha0) + 2 * total

    def setEverettFunction(self, func: Callable) -> None:
        self.everett = func

    def showEverettFunction(self, fig: plt.Figure) -> None:
        ax = fig.add_subplot(111, projection='3d')
        Z = self.everett(self.gridX, self.gridY)
        ax.plot_surface(self.gridX, self.gridY, Z, cmap='viridis')
        ax.set_title('Everett Function')
        ax.set_xlabel('β')
        ax.set_ylabel('α')
        ax.set_zlabel('E(β, α)')
        plt.show()

    def animateHysteresis(self) -> animation.FuncAnimation:
        u_vals = self.historyU[1:]
        out_vals = self.historyOut
        interfaces_x = self.historyInterfaceX
        interfaces_y = self.historyInterfaceY

        if len(u_vals) != len(out_vals) or len(u_vals) != len(interfaces_x):
            raise ValueError("History length mismatch in animation.")

        frames = len(u_vals)
        t_vals = np.arange(len(u_vals))

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        (ax1, ax2), (ax3, ax4) = axs

        # --- Top-left: Input vs time
        ax1.plot(t_vals, u_vals, 'b-', linewidth=1, label='u(t)')
        ax1.set_xlim(0, len(t_vals) - 1)
        ax1.set_ylim(-self.alpha0 * 1.1, self.alpha0 * 1.1)
        ax1.set_xlabel('Time step')
        ax1.set_ylabel('Input u(t)')
        ax1.grid(True)
        line1, = ax1.plot([], [], 'ro', markersize=6)

        # --- Top-right: Output vs time
        ax2.plot(t_vals, out_vals, 'g-', linewidth=1, label='f(t)')
        ax2.set_xlim(0, len(t_vals) - 1)
        ax2.set_ylim(np.min(out_vals) * 1.1, np.max(out_vals) * 1.1)
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('Output f(t)')
        ax2.grid(True)
        line2, = ax2.plot([], [], 'ro', markersize=6)

        # --- Bottom-left: Hysteresis loop (f vs u)
        ax3.plot(u_vals, out_vals, 'm-', linewidth=1)
        ax3.set_xlabel('Input u(t)')
        ax3.set_ylabel('Output f(t)')
        ax3.set_title('Hysteresis Loop')
        ax3.grid(True)
        line3, = ax3.plot([], [], 'ro', markersize=6)

        # --- Bottom-right: Preisach triangle
        tri_x = [-self.beta0, self.beta0, -self.beta0, -self.beta0]
        tri_y = [-self.alpha0, self.alpha0, self.alpha0, -self.alpha0]
        ax4.plot(tri_x, tri_y, 'k-', linewidth=2)
        line4, = ax4.plot([], [], 'r-', linewidth=2, label='L(t)')
        ax4.set_xlim(-self.beta0 * 1.1, self.beta0 * 1.1)
        ax4.set_ylim(-self.alpha0 * 1.1, self.alpha0 * 1.1)
        ax4.set_xlabel('β')
        ax4.set_ylabel('α')
        ax4.set_aspect('equal')
        ax4.grid(True)
        ax4.set_title('Preisach Plane')
        ax4.legend(loc='lower right')

        def update_line(num):
            # Update time-series dots
            line1.set_data([t_vals[num]], [u_vals[num]])
            line2.set_data([t_vals[num]], [out_vals[num]])
            # Update hysteresis dot
            line3.set_data([u_vals[num]], [out_vals[num]])
            # Update staircase
            line4.set_data(interfaces_x[num], interfaces_y[num])
            return line1, line2, line3, line4

        anim = animation.FuncAnimation(
            fig,
            update_line,
            frames=frames,
            interval=30,
            blit=True,
            repeat=False
        )

        plt.tight_layout()
        plt.show()
        return anim

