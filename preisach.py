import numpy as np
from scipy.interpolate import RectBivariateSpline

class PreisachModel:
    """
    Classical scalar Preisach hysteresis model.
    Based on TU Darmstadt Lecture 7 (SS 2025), slides 63–64, 68.
    """

    def __init__(self, u_range=(-1.0, 1.0), n_ab=80, gamma=1.0):
        self.u_min, self.u_max = u_range
        self.n_ab = n_ab
        self.gamma = gamma

        # Grids
        self.alpha_grid = np.linspace(self.u_min, self.u_max, n_ab)
        self.beta_grid  = np.linspace(self.u_min, self.u_max, n_ab)
        self.alpha_mesh, self.beta_mesh = np.meshgrid(
            self.alpha_grid, self.beta_grid, indexing='ij')

        # Everett function F(α,β) = γ(α−β)²/2
        self.everett = self.gamma * (self.alpha_mesh - self.beta_mesh) ** 2 / 2
        self.everett[self.alpha_mesh < self.beta_mesh] = 0.0
        self._F_spline = RectBivariateSpline(
            self.alpha_grid, self.beta_grid, self.everett, kx=1, ky=1)

        self.clear_history()

    def _F(self, alpha, beta):
        """Everett function F(α, β)."""
        if alpha <= beta:
            return 0.0
        return float(self._F_spline(alpha, beta))

    def clear_history(self):
        """Start from negative saturation: all relays down."""
        # Turning points: [β₀, α₁, β₁, α₂, β₂, ...]
        self.turning_points = [self.u_min]

    def __call__(self, u):
        """Update state with new input u and return output f."""
        self._update_memory_line(u)
        return self._compute_output()

    def _update_memory_line(self, u):
        tp = self.turning_points
        if len(tp) == 0:
            tp.append(self.u_min)

        # Wiping-out property
        while len(tp) >= 2:
            if len(tp) % 2 == 0:  # last move: down → now going up?
                if u <= tp[-1]:
                    break
                # Remove all (α_prev, β_prev) with α_prev <= u
                while len(tp) >= 2 and tp[-2] <= u:
                    tp.pop()  # β
                    tp.pop()  # α
                break
            else:  # last move: up → now going down?
                if u >= tp[-1]:
                    break
                # Remove all (α_prev, β_prev) with β_prev >= u
                while len(tp) >= 2 and tp[-2] >= u:
                    tp.pop()  # α
                    tp.pop()  # β
                break

        # Add new turning point if direction changes
        if len(tp) == 1:
            if u > tp[-1]:
                tp.append(u)
        else:
            last_was_up = (len(tp) % 2 == 1)  # odd length → last segment up
            if (last_was_up and u < tp[-1]) or (not last_was_up and u > tp[-1]):
                tp.append(u)

    def _compute_output(self):
        tp = self.turning_points
        u = tp[-1]  # current input

        # Full triangle integral (negative saturation)
        f = -self._F(self.u_max, self.u_min)

        # Sum over completed rectangles
        for k in range(1, len(tp)):
            if k % 2 == 1:  # tp[k] is α_k (upward turning point)
                alpha_k = tp[k]
                beta_km1 = tp[k - 1]
                f += 2 * self._F(alpha_k, beta_km1)
            else:  # tp[k] is β_k (downward turning point)
                alpha_km1 = tp[k - 1]
                beta_k = tp[k]
                f -= 2 * self._F(alpha_km1, beta_k)

        return f

    def triangle_mask(self):
        """Generate mask for Preisach triangle visualization."""
        n = self.n_ab
        mask = np.zeros((n, n))

        tp = self.turning_points[:]
        if len(tp) == 1:
            # All relays down → S− everywhere
            mask[:] = -1
        else:
            # Fill S− regions (relays = -1)
            for i in range(0, len(tp) - 1, 2):
                beta_i = tp[i]
                alpha_ip1 = tp[i + 1] if i + 1 < len(tp) else tp[-1]
                i_alpha = np.searchsorted(self.alpha_grid, alpha_ip1, side='right')
                j_beta = np.searchsorted(self.beta_grid, beta_i, side='left')
                mask[i_alpha:, :j_beta] = -1

            # Fill S+ regions (relays = +1)
            for i in range(1, len(tp) - 1, 2):
                alpha_i = tp[i]
                beta_ip1 = tp[i + 1] if i + 1 < len(tp) else tp[-1]
                i_alpha = np.searchsorted(self.alpha_grid, alpha_i, side='left')
                j_beta = np.searchsorted(self.beta_grid, beta_ip1, side='right')
                mask[:i_alpha, j_beta:] = 1

            # Handle final incomplete segment
            if len(tp) % 2 == 1:  # last segment: increasing → S+ for α ≤ u, β ≥ last_min
                u_curr = tp[-1]
                last_min = tp[-2]
                i_alpha = np.searchsorted(self.alpha_grid, u_curr, side='right')
                j_beta = np.searchsorted(self.beta_grid, last_min, side='left')
                mask[:i_alpha, j_beta:] = 1
            else:  # last segment: decreasing → S− for α ≥ last_max, β ≤ u
                u_curr = tp[-1]
                last_max = tp[-2]
                i_alpha = np.searchsorted(self.alpha_grid, last_max, side='left')
                j_beta = np.searchsorted(self.beta_grid, u_curr, side='right')
                mask[i_alpha:, :j_beta] = -1

        # Enforce α ≥ β (triangle domain)
        for i in range(n):
            for j in range(i):
                if self.alpha_grid[i] < self.beta_grid[j]:
                    mask[i, j] = 0

        extent = [self.u_min, self.u_max, self.u_min, self.u_max]
        return mask, extent