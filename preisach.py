# preisach.py
import numpy as np

class PreisachModel:
    """
    Scalar Preisach model with uniform density μ(α,β) = γ.
    Implements the exact formula from TU Darmstadt Lecture 7, Slide 64.
    """

    def __init__(self, u_range=(-1.0, 1.0), gamma=1.0):
        self.u_min, self.u_max = u_range
        if self.u_min >= self.u_max:
            raise ValueError("u_min must be < u_max")
        self.gamma = gamma
        self.clear_history()

    def _F(self, alpha, beta):
        """Everett function F(α,β) = γ(α−β)² / 2 for α ≥ β, else 0."""
        if alpha <= beta:
            return 0.0
        return 0.5 * self.gamma * (alpha - beta) ** 2

    def clear_history(self):
        """
        Start from negative saturation.
        Memory line: list of turning points [β₀, α₁, β₁, α₂, β₂, ...]
        We always keep β₀ = u_min as the first point.
        """
        self.turning_points = [self.u_min]  # β₀

    def __call__(self, u):
        # Clamp input to physical bounds (optional but safe)
        u = max(self.u_min, min(self.u_max, u))
        tp = self.turning_points

        # --- Wiping-out property ---
        if len(tp) >= 2:
            if len(tp) % 2 == 0:
                # Last move was DOWN (tp ends with α_k), now possibly increasing
                if u > tp[-1]:
                    # Wipe out all (α_i, β_i) with α_i <= u, but keep β₀
                    while len(tp) > 2 and tp[-2] <= u:
                        tp.pop()  # β_i
                        tp.pop()  # α_i
            else:
                # Last move was UP (tp ends with β_k), now possibly decreasing
                if u < tp[-1]:
                    # Wipe out all (α_i, β_i) with β_i >= u
                    while len(tp) > 2 and tp[-2] >= u:
                        tp.pop()  # α_i
                        tp.pop()  # β_i

        # --- Add new turning point only on reversal ---
        if len(tp) == 1:
            # Only β₀ exists → if moving up, add first α₁
            if u > tp[-1]:
                tp.append(u)
        else:
            # Determine current direction from last segment
            # Even length → last move was UP (ended at α) → now decreasing
            # Odd length  → last move was DOWN (ended at β) → now increasing
            last_was_up = (len(tp) % 2 == 0)  # even → last was up
            last_u = tp[-1]
            if (last_was_up and u < last_u) or (not last_was_up and u > last_u):
                tp.append(u)

        # --- Compute output using Slide 64 formula ---
        f = -self._F(self.u_max, self.u_min)  # -F(α₀, β₀)

        n = len(tp)
        # Sum from k=1 to n-1
        for k in range(1, n):
            if k % 2 == 1:  # tp[k] is α_k (odd index: 1,3,5,...)
                alpha_k = tp[k]
                beta_km1 = tp[k - 1]
                f += 2 * self._F(alpha_k, beta_km1)
            else:  # tp[k] is β_k (even index: 2,4,6,...)
                alpha_km1 = tp[k - 1]
                beta_k = tp[k]
                f -= 2 * self._F(alpha_km1, beta_k)

        # Final term: depends on current direction
        if n % 2 == 1:
            # Odd length → last point is β_n → currently increasing → add 2*F(u, β_n)
            beta_n = tp[-1]
            f += 2 * self._F(u, beta_n)
        else:
            # Even length → last point is α_n → currently decreasing → subtract 2*F(α_n, u)
            alpha_n = tp[-1]
            f -= 2 * self._F(alpha_n, u)

        return f

    def triangle_mask(self):
        """Dummy placeholder for animation (optional)."""
        n = 80
        mask = -np.ones((n, n))
        extent = [self.u_min, self.u_max, self.u_min, self.u_max]
        return mask, extent