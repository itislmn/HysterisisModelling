import numpy as np
from scipy.interpolate import RectBivariateSpline

class PreisachModel:
    """
    Classical scalar Preisach hysteresis model.
    ------------------------------------------
    Parameters
    ----------
    u_range : (u_min, u_max)
        Range of the input signal (used for triangle limits).
    n_ab    : int
        Discretisation steps for the triangle (α,β) grids.
    gamma   : float
        Parameter of the simple Everett function
        F(α,β)=γ(α-β)²/2  ->  μ(α,β)=γ
    """

    def __init__(self, u_range=(-1.0, 1.0), n_ab=80, gamma=1.0):
        self.u_min, self.u_max = u_range
        self.n_ab = n_ab
        self.gamma = gamma

        # Build α, β grids (α ≥ β)
        self.alpha_grid = np.linspace(self.u_min, self.u_max, n_ab)
        self.beta_grid  = np.linspace(self.u_min, self.u_max, n_ab)
        self.alpha_mesh, self.beta_mesh = np.meshgrid(
            self.alpha_grid, self.beta_grid, indexing='ij')

        # Everett function F(α,β) = γ(α−β)²/2 on the triangle
        self.everett = self.gamma * (self.alpha_mesh - self.beta_mesh) ** 2 / 2
        self.everett[self.alpha_mesh < self.beta_mesh] = 0.0
        self._F_spline = RectBivariateSpline(
            self.alpha_grid, self.beta_grid, self.everett, kx=1, ky=1)

        self.clear_history()

    def clear_history(self):
        """Reset to negative saturation: all relays down."""
        self.turning_points = [self.u_min]  # Start from negative saturation

    def __call__(self, u):
        """Update with new input u and return output f."""
        self._update_memory_line(u)
        return self._compute_output()

    def _F(self, alpha, beta):
        """Everett function F(α, β)."""
        if alpha <= beta:
            return 0.0
        return float(self._F_spline(alpha, beta))

    def _update_memory_line(self, u):
        """Apply wiping-out property and update turning points."""
        tp = self.turning_points

        # Wipe-out: remove points that are "overwritten"
        while len(tp) >= 2:
            if len(tp) % 2 == 0:  # last segment was decreasing → next should be increasing
                if u <= tp[-1]:
                    break
                # Wipe all maxima ≤ u
                while len(tp) >= 2 and tp[-2] <= u:
                    tp.pop()
                    tp.pop()
                break
            else:  # last segment was increasing → next should be decreasing
                if u >= tp[-1]:
                    break
                # Wipe all minima ≥ u
                while len(tp) >= 2 and tp[-2] >= u:
                    tp.pop()
                    tp.pop()
                break

        # Add new turning point if input reverses
        if len(tp) == 1:
            if u > tp[-1]:
                tp.append(u)
        else:
            if (len(tp) % 2 == 0 and u > tp[-1]) or (len(tp) % 2 == 1 and u < tp[-1]):
                tp.append(u)

    def _compute_output(self):
        """Compute f(t) = -F(α₀,β₀) + 2 Σ ΔF over staircase."""
        tp = self.turning_points
        f = -self._F(self.u_max, self.u_min)  # negative saturation value

        # Sum over completed rectangles
        for i in range(1, len(tp)):
            if i % 2 == 1:  # odd index → upward transition
                f += 2 * self._F(tp[i], tp[i - 1])
            else:  # even index → downward transition
                f -= 2 * self._F(tp[i], tp[i - 1])

        return f

    def triangle_mask(self):
        """
        Return mask for visualization:
        +1 = S+ (relays up), -1 = S− (relays down), 0 = neutral.
        """
        n = self.n_ab
        mask = np.zeros((n, n))

        tp = self.turning_points[:]
        if len(tp) == 1:
            # All relays down
            mask[:] = -1
        else:
            # Fill S− regions (relays down)
            for i in range(0, len(tp) - 1, 2):
                α_low = tp[i]
                β_high = tp[i + 1] if i + 1 < len(tp) else tp[-1]
                iα = np.searchsorted(self.alpha_grid, α_low, side='left')
                jβ = np.searchsorted(self.beta_grid, β_high, side='right')
                mask[iα:, :jβ] = -1

            # Fill S+ regions (relays up)
            for i in range(1, len(tp) - 1, 2):
                α_high = tp[i]
                β_low = tp[i + 1] if i + 1 < len(tp) else tp[-1]
                iα = np.searchsorted(self.alpha_grid, α_high, side='right')
                jβ = np.searchsorted(self.beta_grid, β_low, side='left')
                mask[:iα, jβ:] = 1

            # Final segment
            if len(tp) % 2 == 1:
                # Last segment increasing → region α ≤ u, β ≥ last_min is up
                u = tp[-1]
                iα = np.searchsorted(self.alpha_grid, u, side='right')
                jβ = np.searchsorted(self.beta_grid, tp[-2], side='left')
                mask[:iα, jβ:] = 1
            else:
                # Last segment decreasing → region α ≥ last_max, β ≤ u is down
                u = tp[-1]
                iα = np.searchsorted(self.alpha_grid, tp[-2], side='left')
                jβ = np.searchsorted(self.beta_grid, u, side='right')
                mask[iα:, :jβ] = -1

        # Enforce α ≥ β (triangle domain)
        for i in range(n):
            for j in range(n):
                if self.alpha_grid[i] < self.beta_grid[j]:
                    mask[i, j] = 0

        extent = [self.u_min, self.u_max, self.u_min, self.u_max]
        return mask, extent