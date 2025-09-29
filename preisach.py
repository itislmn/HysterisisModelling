# preisach.py
import numpy as np
from scipy.interpolate import RectBivariateSpline

class PreisachModel:
    def __init__(self, u_range=(-1.0, 1.0), n_ab=80, gamma=1.0):
        self.u_min, self.u_max = u_range
        self.n_ab = n_ab
        self.gamma = gamma

        self.alpha_grid = np.linspace(self.u_min, self.u_max, n_ab)
        self.beta_grid  = np.linspace(self.u_min, self.u_max, n_ab)
        self.alpha_mesh, self.beta_mesh = np.meshgrid(
            self.alpha_grid, self.beta_grid, indexing='ij')

        self.everett = self.gamma * (self.alpha_mesh - self.beta_mesh) ** 2 / 2
        self.everett[self.alpha_mesh < self.beta_mesh] = 0.0
        self._F_spline = RectBivariateSpline(
            self.alpha_grid, self.beta_grid, self.everett, kx=1, ky=1)

        self.clear_history()

    def _F(self, alpha, beta):
        if alpha <= beta:
            return 0.0
        return float(self._F_spline(alpha, beta))

    def clear_history(self):
        # Always start with one point: negative saturation
        self.turning_points = [self.u_min]

    def __call__(self, u):
        self._update_memory_line(u)
        return self._compute_output()

    def _update_memory_line(self, u):
        tp = self.turning_points

        # --- Wiping-out ---
        while len(tp) >= 2:
            if len(tp) % 2 == 0:  # last move was down → now going up?
                if u <= tp[-1]:
                    break
                # Remove pairs while previous max <= u
                while len(tp) >= 2 and tp[-2] <= u:
                    tp.pop()  # β
                    tp.pop()  # α
                    # After popping, ensure we don't go below 1 element
                    if len(tp) < 1:
                        tp.append(self.u_min)
                break
            else:  # last move was up → now going down?
                if u >= tp[-1]:
                    break
                while len(tp) >= 2 and tp[-2] >= u:
                    tp.pop()  # α
                    tp.pop()  # β
                    if len(tp) < 1:
                        tp.append(self.u_min)
                break

        # --- Add new turning point on reversal ---
        if len(tp) == 1:
            if u > tp[-1]:
                tp.append(u)
        else:
            # Safe access: tp[-1] exists because len(tp) >= 2
            last_was_up = (len(tp) % 2 == 1)
            if (last_was_up and u < tp[-1]) or (not last_was_up and u > tp[-1]):
                tp.append(u)

    def _compute_output(self):
        tp = self.turning_points
        f = -self._F(self.u_max, self.u_min)
        for k in range(1, len(tp)):
            if k % 2 == 1:  # α_k
                f += 2 * self._F(tp[k], tp[k - 1])
            else:  # β_k
                f -= 2 * self._F(tp[k - 1], tp[k])
        return f

    def triangle_mask(self):
        n = self.n_ab
        mask = np.zeros((n, n))
        tp = self.turning_points[:]

        if len(tp) == 1:
            mask[:] = -1
        else:
            # Fill S−
            for i in range(0, len(tp) - 1, 2):
                beta_i = tp[i]
                alpha_ip1 = tp[i + 1] if i + 1 < len(tp) else tp[-1]
                i_a = np.searchsorted(self.alpha_grid, alpha_ip1, side='right')
                j_b = np.searchsorted(self.beta_grid, beta_i, side='left')
                mask[i_a:, :j_b] = -1

            # Fill S+
            for i in range(1, len(tp) - 1, 2):
                alpha_i = tp[i]
                beta_ip1 = tp[i + 1] if i + 1 < len(tp) else tp[-1]
                i_a = np.searchsorted(self.alpha_grid, alpha_i, side='left')
                j_b = np.searchsorted(self.beta_grid, beta_ip1, side='right')
                mask[:i_a, j_b:] = 1

            # Final segment
            if len(tp) % 2 == 1:
                u_curr = tp[-1]
                last_min = tp[-2]
                i_a = np.searchsorted(self.alpha_grid, u_curr, side='right')
                j_b = np.searchsorted(self.beta_grid, last_min, side='left')
                mask[:i_a, j_b:] = 1
            else:
                u_curr = tp[-1]
                last_max = tp[-2]
                i_a = np.searchsorted(self.alpha_grid, last_max, side='left')
                j_b = np.searchsorted(self.beta_grid, u_curr, side='right')
                mask[i_a:, :j_b] = -1

        # Enforce α ≥ β
        for i in range(n):
            for j in range(n):
                if self.alpha_grid[i] < self.beta_grid[j]:
                    mask[i, j] = 0

        extent = [self.u_min, self.u_max, self.u_min, self.u_max]
        return mask, extent