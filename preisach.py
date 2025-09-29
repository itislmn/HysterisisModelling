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

        # build α,β grids (α ≥ β)
        self.alpha_grid = np.linspace(self.u_min, self.u_max, n_ab)
        self.beta_grid  = np.linspace(self.u_min, self.u_max, n_ab)
        self.alpha_mesh, self.beta_mesh = np.meshgrid(
            self.alpha_grid, self.beta_grid, indexing='ij')

        # Everett function F(α,β) = γ(α-β)²/2  on the triangle
        self.everett = self.gamma * (self.alpha_mesh - self.beta_mesh) ** 2 / 2
        self.everett[self.alpha_mesh < self.beta_mesh] = 0.0
        self._F_spline = RectBivariateSpline(
            self.alpha_grid, self.beta_grid, self.everett, kx=1, ky=1)

        # initial memory line: negative saturation
        self.clear_history()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def clear_history(self):
        """Reset to negative saturation state."""
        self.L_alpha = [self.u_max]
        self.L_beta  = [self.u_min]

    def __call__(self, u):
        """
        Update model with new input value u and return output f.
        """
        self._update_memory_line(u)
        return self._compute_output()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _F(self, alpha, beta):
        """Everett integral over triangle T(alpha,beta)."""
        if alpha < beta:
            return 0.0
        return float(self._F_spline(alpha, beta))

    def _update_memory_line(self, u):
        """Apply wiping-out and congruency rules."""
        # wipe-out
        while len(self.L_alpha) > 1 and u >= self.L_alpha[-1]:
            self.L_alpha.pop()
            self.L_beta.pop()
        while len(self.L_alpha) > 1 and u <= self.L_beta[-1]:
            self.L_alpha.pop()
            self.L_beta.pop()
        # add new corner
        if u > self.L_alpha[-1]:
            self.L_alpha.append(u)
            self.L_beta.append(self.L_beta[-1])
        elif u < self.L_beta[-1]:
            self.L_beta.append(u)
            self.L_alpha.append(self.L_alpha[-1])
        # else: on horizontal/vertical segment -> nothing to do

    def _compute_output(self):
        """Evaluate f(t) from current memory line."""
        f = -self._F(self.u_max, self.u_min)
        for k in range(1, len(self.L_alpha)):
            f += 2 * (self._F(self.L_alpha[k], self.L_beta[k-1]) -
                      self._F(self.L_alpha[k], self.L_beta[k]))
        return f

    # ------------------------------------------------------------------
    # helpers for visualisation
    # ------------------------------------------------------------------
    def triangle_mask(self):
        """
        Return (mask, extent) usable with imshow:
        mask ==  1  -> S+  (blue)
        mask == -1  -> S-  (red)
        mask ==  0  -> neutral (white)
        extent = [β_min, β_max, α_min, α_max]
        """
        n = self.n_ab
        mask = np.zeros((n, n))
        for k in range(1, len(self.L_alpha)):
            α1, β1 = self.L_alpha[k-1], self.L_beta[k-1]
            α2, β2 = self.L_alpha[k],   self.L_beta[k]
            # rectangular region in triangle coordinates
            i1 = np.searchsorted(self.alpha_grid, α1)
            j1 = np.searchsorted(self.beta_grid,  β1)
            i2 = np.searchsorted(self.alpha_grid, α2)
            j2 = np.searchsorted(self.beta_grid,  β2)
            mask[i2:i1, j1:j2] = -1      # S-
        # last segment
        α_last, β_last = self.L_alpha[-1], self.L_beta[-1]
        i_last = np.searchsorted(self.alpha_grid, α_last)
        j_last = np.searchsorted(self.beta_grid,  β_last)
        mask[i_last:, :j_last] = 1       # S+
        extent = [self.u_min, self.u_max, self.u_min, self.u_max]
        return mask, extent