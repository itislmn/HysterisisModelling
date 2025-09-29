# preisach.py
import numpy as np

class PreisachModel:
    def __init__(self, u_range=(-1.0, 1.0), gamma=1.0):
        self.u_min, self.u_max = u_range
        self.gamma = gamma
        self.clear_history()

    def _F(self, a, b):
        return 0.5 * self.gamma * (a - b)**2 if a > b else 0.0

    def clear_history(self):
        self.turning_points = [self.u_min]  # start at negative saturation

    def __call__(self, u):
        tp = self.turning_points

        # Wiping-out
        while len(tp) >= 2:
            if len(tp) % 2 == 0:  # last segment down -> now up?
                if u <= tp[-1]:
                    break
                while len(tp) >= 2 and tp[-2] <= u:
                    tp.pop()
                    tp.pop()
                break
            else:  # last segment up -> now down?
                if u >= tp[-1]:
                    break
                while len(tp) >= 2 and tp[-2] >= u:
                    tp.pop()
                    tp.pop()
                break

        # Add new turning point if direction changes
        if len(tp) == 1:
            if u > tp[-1]:
                tp.append(u)
        else:
            last_up = (len(tp) % 2 == 1)
            if (last_up and u < tp[-1]) or (not last_up and u > tp[-1]):
                tp.append(u)

        # Compute output using slide 64 formula
        f = -self._F(self.u_max, self.u_min)
        for k in range(1, len(tp)):
            if k % 2 == 1:  # α_k
                f += 2 * self._F(tp[k], tp[k-1])
            else:  # β_k
                f -= 2 * self._F(tp[k-1], tp[k])

        # FINAL TERM: current u is tp[-1], so no extra term needed
        # Because we added u as a turning point, the sum is complete.
        return f

    def triangle_mask(self):
        n = 80
        mask = -np.ones((n, n))
        extent = [self.u_min, self.u_max, self.u_min, self.u_max]
        return mask, extent