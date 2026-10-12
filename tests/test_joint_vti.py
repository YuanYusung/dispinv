"""Numerical regression tests for restricted VTI joint inversion."""
import unittest

import numpy as np

from BFGS_joint_VTI import (
    invert_joint_bfgs,
    love_phase_vti,
    rayleigh_phase_vti,
)
from disba import PhaseDispersion


class TestVTIForward(unittest.TestCase):
    def test_single_layer_analytic_love(self):
        """The SH surface traction must satisfy the exact one-layer tan equation."""
        h = np.array([0.10, 0.0])
        vs = np.array([0.8, 2.0])
        rho = np.array([2.0, 2.3])
        gamma = np.array([0.05, 0.02])
        p = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.70, 1.0])
        c = love_phase_vti(p, h, vs, gamma, rho, scan_points=120)
        self.assertTrue(np.all(np.diff(c) > 0))
        vsh = vs * (1.0 + gamma)
        self.assertTrue(np.all(c > vsh[0]))
        self.assertTrue(np.all(c < vsh[-1]))
        L, N = rho * vs**2, rho * vsh**2
        for period, speed in zip(p, c):
            om = 2 * np.pi / period
            k = om / speed
            q = np.sqrt((rho[0] * om**2 - N[0] * k**2) / L[0])
            a = np.sqrt((N[1] * k**2 - rho[1] * om**2) / L[1])
            left = L[0] * q * np.tan(q * h[0])
            right = L[1] * a
            self.assertLess(abs(left - right) / max(1, abs(left), abs(right)), 2e-7)

    def test_isotropic_limit_matches_disba(self):
        """With gamma=0 the independent SH implementation must match disba Love."""
        h = np.array([0.09, 0.16, 0.0])
        vs = np.array([0.75, 1.20, 2.10])
        rho = np.array([2.0, 2.1, 2.3])
        p = np.linspace(0.055, 0.38, 12)
        observed = PhaseDispersion(h, 2 * vs, vs, rho)(
            p, mode=0, wave="love"
        )
        self.assertEqual(len(observed.velocity), len(p))
        ours = love_phase_vti(p, h, vs, np.zeros_like(vs), rho, scan_points=160)
        np.testing.assert_allclose(ours, observed.velocity, rtol=2e-3, atol=2e-3)

    def test_anisotropy_affects_love_but_not_restricted_rayleigh(self):
        h = np.array([0.10, 0.0])
        vs = np.array([0.8, 2.0])
        rho = np.array([2.0, 2.3])
        p = np.linspace(0.075, 0.5, 7)
        c0 = love_phase_vti(p, h, vs, np.zeros(2), rho)
        c1 = love_phase_vti(p, h, vs, np.array([0.06, 0.02]), rho)
        self.assertGreater(np.max(abs(c1 - c0)), 0.01)
        r0 = rayleigh_phase_vti(p, h, vs, rho)
        r1 = rayleigh_phase_vti(p, h, vs, rho)
        np.testing.assert_allclose(r0, r1, atol=1e-9)


class TestJointRecovery(unittest.TestCase):
    def test_two_layer_synthetic_fit(self):
        """Noise-free end-to-end R/L recovery; checks solution fit, not uniqueness."""
        h = np.array([0.12, 0.0])
        rho = np.array([2.0, 2.3])
        vtrue = np.array([0.8, 2.0])
        gtrue = np.array([0.06, 0.02])
        p = np.linspace(0.075, 0.50, 7)
        rayleigh = rayleigh_phase_vti(p, h, vtrue, rho)
        love = love_phase_vti(p, h, vtrue, gtrue, rho)
        res, model = invert_joint_bfgs(
            p, rayleigh, np.full_like(p, 0.01),
            p, love, np.full_like(p, 0.01),
            h, np.array([0.85, 1.94]), np.array([0.0, 0.0]), rho,
            lambda_gamma=0.0, lambda_smooth=0.0, lambda_vs=0.0,
            maxiter=70, scan_points=60
        )
        print("Recovered Vsv:", model["vsv_km_s"], "true:", vtrue)
        print("Recovered gamma:", model["gamma"], "true:", gtrue)
        np.testing.assert_allclose(model["rayleigh_pred"], rayleigh, atol=0.004)
        np.testing.assert_allclose(model["love_pred"], love, atol=0.004)
        self.assertLess(np.max(abs(model["vsv_km_s"] - vtrue)), 0.10)
        self.assertLess(np.max(abs(model["gamma"] - gtrue)), 0.06)
        self.assertTrue(np.isfinite(res.fun))


if __name__ == "__main__":
    unittest.main()
