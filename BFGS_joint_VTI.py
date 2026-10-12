"""BFGS inversion of fundamental Rayleigh + Love dispersion for restricted VTI.

Model: A=C=rho*Vp**2, F=A-2*L, L=rho*Vsv**2, N=rho*Vsh**2,
with Vsh=Vsv*(1+gamma).  Vp/Vsv, rho and finite-layer thickness are fixed.

Rayleigh is the isotropic P-SV subsystem (computed by disba); Love is
the *VTI* SH subsystem (L and N enter independently).  This is NOT a
general five-constant VTI inversion.

CSV curve: period_s,c_km_s[,sigma_km_s] (frequency_hz instead of period_s
is supported). Model: thickness_km,vsv_km_s[,rho_g_cm3,gamma_init].
Last model row is a half-space with thickness 0. Units: km, km/s, seconds.

Examples:
  python BFGS_joint_VTI.py --demo --output-dir demo_vti
  python BFGS_joint_VTI.py --rayleigh rayleigh.csv --love love.csv \
      --model initial_model.csv --output-dir vti_result
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import expit, logit


def validate_model(h, vs, gamma, rho):
    h, vs, gamma, rho = map(lambda x: np.asarray(x, dtype=float), (h, vs, gamma, rho))
    if h.ndim != 1 or len(h) < 2 or any(x.shape != h.shape for x in (vs, gamma, rho)):
        raise ValueError("Model arrays must be 1-D, with equal lengths >=2")
    if any(not np.all(np.isfinite(x)) for x in (h, vs, gamma, rho)):
        raise ValueError("Nonfinite model parameters")
    if np.any(h[:-1] <= 0) or not np.isclose(h[-1], 0):
        raise ValueError("Positive finite-layer thickness and zero-thickness half-space required")
    if np.any(vs <= 0) or np.any(1 + gamma <= 0) or np.any(rho <= 0):
        raise ValueError("Vsv, Vsh and density must be positive")
    return h, vs, gamma, rho


def love_secular(c, omega, h, L, N, rho):
    """Normalized surface SH traction from the decaying half-space solution.

    The VTI SH equation is (L u')' + (rho*omega**2-N*k**2)*u=0,
    with k=omega/c. Propagation upward preserves signs (no poles).
    """
    k = omega / c
    bottom = N[-1] * k*k - rho[-1] * omega*omega
    if bottom <= 0:
        return np.nan
    u, t = 1., -np.sqrt(L[-1] * bottom)
    for j in range(len(h)-2, -1, -1):
        q2 = (N[j]*k*k - rho[j]*omega*omega) / L[j]
        qh = np.sqrt(abs(q2))*h[j]
        if q2 > 0:
            q = np.sqrt(q2)
            if qh > 35:
                e = np.exp(-2*qh)
                ch, sh = (1+e)/2, (1-e)/2  # exp(-qh) scaling
            else:
                ch, sh = np.cosh(qh), np.sinh(qh)
            b, d = sh/q, q*sh
        elif q2 < 0:
            q = np.sqrt(-q2)
            ch, b, d = np.cos(qh), np.sin(qh)/q, -q*np.sin(qh)
        else:
            ch, b, d = 1., h[j], 0.
        u, t = ch*u - b*t/L[j], ch*t - L[j]*d*u
        norm = np.hypot(u, t / (rho[j]*omega*np.sqrt(N[j]/rho[j])))
        if not np.isfinite(norm) or norm == 0:
            raise FloatingPointError("Unstable VTI SH propagator")
        u, t = u/norm, t/norm
    return t / np.hypot(t, rho[0]*omega*np.sqrt(N[0]/rho[0])*u)


def love_phase_vti(periods, h, vsv, gamma, rho, scan_points=80):
    """Return fundamental *trapped* VTI Love-wave phase velocity, km/s.

    Raise when no real, half-space-evanescent Love mode is found.
    """
    h, vsv, gamma, rho = validate_model(h, vsv, gamma, rho)
    periods = np.asarray(periods, dtype=float)
    if periods.ndim != 1 or not np.all(np.isfinite(periods)) or np.any(periods <= 0):
        raise ValueError("periods must be a positive finite 1-D array")
    if scan_points < 16:
        raise ValueError("scan_points must be >=16")
    vsh = vsv*(1+gamma)
    L, N = rho*vsv*vsv, rho*vsh*vsh
    low, high = np.min(vsh[:-1])*(1-1e-9), vsh[-1]*(1-1e-10)
    if low >= high:
        raise ValueError("No guided Love mode: half-space Vsh is too slow")
    s = np.linspace(0, 1, scan_points)
    grid = low + (high-low)*0.5*(1-np.cos(np.pi*s))
    out = np.empty(len(periods))
    for i, period in enumerate(periods):
        omega = 2*np.pi/period
        last_c = grid[0]
        last_value = love_secular(last_c, omega, h, L, N, rho)
        for c in grid[1:]:
            value = love_secular(c, omega, h, L, N, rho)
            if np.isfinite(last_value) and np.isfinite(value) and last_value*value <= 0:
                out[i] = brentq(love_secular, last_c, c, args=(omega, h, L, N, rho),
                                xtol=1e-11, rtol=1e-12)
                break
            last_c, last_value = c, value
        else:
            raise ValueError(f"Love fundamental mode unavailable at T={period:.6g} s")
    return out


def rayleigh_phase_vti(periods, h, vsv, rho, vpvs=2.0):
    """Rayleigh phase speeds in restricted VTI: P-SV part is isotropic."""
    from disba import PhaseDispersion
    p = np.asarray(periods, dtype=float)
    if p.ndim != 1 or np.any(p <= 0) or not np.all(np.isfinite(p)):
        raise ValueError("periods must be positive finite 1-D array")
    if vpvs <= np.sqrt(2):
        raise ValueError("vpvs must exceed sqrt(2)")
    order = np.argsort(p)
    pred = PhaseDispersion(np.asarray(h), vpvs*np.asarray(vsv),
                           np.asarray(vsv), np.asarray(rho))(
                               p[order], mode=0, wave="rayleigh")
    if len(pred.period) != len(p) or not np.allclose(pred.period, p[order],
                                                     atol=1e-10, rtol=1e-7):
        raise ValueError("Rayleigh fundamental mode missing at one or more periods")
    out = np.empty_like(p)
    out[order] = pred.velocity
    return out


def invert_joint_bfgs(period_r, c_r, sigma_r, period_l, c_l, sigma_l,
                      thickness, vsv0, gamma0, rho, vpvs=2.,
                      vs_bounds=(.1, 3.5), gamma_bounds=(-.15, .15),
                      lambda_gamma=.01, lambda_smooth=.03, lambda_vs=.01,
                      maxiter=150, fd_step=1e-3, scan_points=80):
    """Jointly invert Vsv and gamma with smooth bounds and true BFGS."""
    h, vsv0, gamma0, rho = validate_model(thickness, vsv0, gamma0, rho)
    pr, cr, sr, pl, cl, sl = map(lambda a: np.asarray(a, dtype=float),
                                 (period_r, c_r, sigma_r, period_l, c_l, sigma_l))
    if (len(pr) < 2 or len(pl) < 2 or
        pr.shape != cr.shape or cr.shape != sr.shape or
        pl.shape != cl.shape or cl.shape != sl.shape or
        any(a.ndim != 1 or np.any(~np.isfinite(a)) or np.any(a <= 0)
            for a in (pr, cr, sr, pl, cl, sl))):
        raise ValueError("Both curves require matching positive observations and sigmas")
    if (vs_bounds[0] <= 0 or vs_bounds[1] <= vs_bounds[0] or
        gamma_bounds[0] <= -1 or gamma_bounds[1] <= gamma_bounds[0]):
        raise ValueError("Invalid bounds")
    if any(w < 0 for w in (lambda_gamma, lambda_smooth, lambda_vs)):
        raise ValueError("Regularization weights must be nonnegative")
    n = len(h)

    def encode(p, bounds):
        q = (p-bounds[0])/(bounds[1]-bounds[0])
        if np.any(q <= 0) or np.any(q >= 1):
            raise ValueError("Initial model must be strictly inside bounds")
        return logit(q)

    def decode(z):
        vsv = vs_bounds[0] + (vs_bounds[1]-vs_bounds[0])*expit(z[:n])
        gamma = gamma_bounds[0] + (gamma_bounds[1]-gamma_bounds[0])*expit(z[n:])
        return vsv, gamma

    z0 = np.r_[encode(vsv0, vs_bounds), encode(gamma0, gamma_bounds)]

    def objective(z):
        vsv, gamma = decode(z)
        try:
            r = rayleigh_phase_vti(pr, h, vsv, rho, vpvs)
            l = love_phase_vti(pl, h, vsv, gamma, rho, scan_points)
            data = .5*(np.mean(((r-cr)/sr)**2)+np.mean(((l-cl)/sl)**2))
        except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
            return 1e5 + np.sum((z-z0)**2)
        reg = lambda_gamma*np.mean((gamma/.1)**2)
        if n > 1:
            reg += lambda_smooth*np.mean((np.diff(gamma)/.1)**2)
        if n > 2:
            reg += lambda_vs*np.mean(np.diff(np.log(vsv), n=2)**2)
        return float(data+reg)

    result = minimize(objective, z0, method="BFGS", jac="3-point",
                      options={"maxiter":maxiter, "gtol":1e-3, "disp":True,
                               "finite_diff_rel_step":fd_step})
    vsv, gamma = decode(result.x)
    model = {
        "thickness_km":h, "vsv_km_s":vsv, "vsh_km_s":vsv*(1+gamma),
        "gamma":gamma, "xi":(1+gamma)**2, "vp_km_s":vpvs*vsv,
        "rho_g_cm3":rho,
        "rayleigh_pred":rayleigh_phase_vti(pr, h, vsv, rho, vpvs),
        "love_pred":love_phase_vti(pl, h, vsv, gamma, rho, scan_points),
    }
    return result, model


def read_csv(path):
    a = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8",
                      dtype=None, ndmin=1)
    if a.size == 0 or a.dtype.names is None:
        raise ValueError(f"Invalid CSV: {path}")
    return a


def load_curve(path, default_sigma):
    a = read_csv(path)
    if "period_s" in a.dtype.names:
        p = np.asarray(a["period_s"], dtype=float)
    elif "frequency_hz" in a.dtype.names:
        p = 1/np.asarray(a["frequency_hz"], dtype=float)
    else:
        raise ValueError(f"{path} requires period_s or frequency_hz")
    if "c_km_s" not in a.dtype.names:
        raise ValueError(f"{path} requires c_km_s")
    sig = (np.asarray(a["sigma_km_s"], dtype=float)
           if "sigma_km_s" in a.dtype.names else np.full(len(a), default_sigma))
    return p, np.asarray(a["c_km_s"], dtype=float), sig


def load_model(path):
    a = read_csv(path)
    if not {"thickness_km", "vsv_km_s"}.issubset(a.dtype.names):
        raise ValueError(f"{path} requires thickness_km,vsv_km_s")
    n = len(a)
    h, vs = np.asarray(a["thickness_km"], dtype=float), np.asarray(a["vsv_km_s"], dtype=float)
    rho = (np.asarray(a["rho_g_cm3"], dtype=float)
           if "rho_g_cm3" in a.dtype.names else np.full(n, 2.))
    gamma = (np.asarray(a["gamma_init"], dtype=float)
             if "gamma_init" in a.dtype.names else np.zeros(n))
    return validate_model(h, vs, gamma, rho)


def save_csv(path, columns, values):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(zip(*values))


def generate_demo(folder, vpvs):
    h, vs, gamma, rho = (np.array([.09, .16, 0.]),
                         np.array([.75, 1.2, 2.1]),
                         np.array([.065, .035, 0.]),
                         np.array([2., 2.1, 2.3]))
    p = np.linspace(.055, .38, 14)
    for wave, c in (("rayleigh", rayleigh_phase_vti(p, h, vs, rho, vpvs)),
                    ("love", love_phase_vti(p, h, vs, gamma, rho))):
        save_csv(folder/f"{wave}_demo.csv", ["period_s","c_km_s","sigma_km_s"],
                 [p, c, np.full(len(p), .01)])
    save_csv(folder/"initial_model_demo.csv",
             ["thickness_km","vsv_km_s","rho_g_cm3","gamma_init"],
             [h, [.86, 1.32, 1.95], rho, [0., 0., 0.]])
    save_csv(folder/"truth_demo.csv", ["thickness_km","vsv_km_s","gamma","xi"],
             [h, vs, gamma, (1+gamma)**2])
    return (folder/"rayleigh_demo.csv", folder/"love_demo.csv",
            folder/"initial_model_demo.csv")


def plot_results(folder, pr, cr, pl, cl, m):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,3,figsize=(13,4))
    for a,p,obs,pred,title in (
        (ax[0], pr, cr, m["rayleigh_pred"], "Rayleigh"),
        (ax[1], pl, cl, m["love_pred"], "Love")):
        ix=np.argsort(p)
        a.plot(p[ix], obs[ix], "ko", label="Observed")
        a.plot(p[ix], pred[ix], "-", label="Predicted")
        a.set(xlabel="Period (s)", ylabel="Phase velocity (km/s)", title=title)
        a.legend()
    h=m["thickness_km"]
    edges=np.r_[0,np.cumsum(h[:-1]),np.sum(h[:-1])+max(.05,.5*np.sum(h[:-1]))]
    ax[2].stairs(100*m["gamma"], edges, baseline=None, orientation="horizontal")
    ax[2].invert_yaxis()
    ax[2].set(xlabel="gamma (%)", ylabel="Depth (km)", title="Radial anisotropy")
    fig.tight_layout()
    fig.savefig(folder/"joint_inversion.png", dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--rayleigh", type=Path)
    parser.add_argument("--love", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("joint_vti_results"))
    parser.add_argument("--vpvs", type=float, default=2.)
    parser.add_argument("--sigma-default", type=float, default=.02)
    parser.add_argument("--vs-min", type=float, default=.1)
    parser.add_argument("--vs-max", type=float, default=3.5)
    parser.add_argument("--gamma-min", type=float, default=-.15)
    parser.add_argument("--gamma-max", type=float, default=.15)
    parser.add_argument("--lambda-gamma", type=float, default=.01)
    parser.add_argument("--lambda-smooth", type=float, default=.03)
    parser.add_argument("--lambda-vs", type=float, default=.01)
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--fd-step", type=float, default=1e-3)
    parser.add_argument("--scan-points", type=int, default=80)
    parser.add_argument("--no-plot", action="store_true")
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.demo:
        args.rayleigh,args.love,args.model=generate_demo(args.output_dir,args.vpvs)
    elif not (args.rayleigh and args.love and args.model):
        parser.error("Provide --rayleigh --love --model or --demo")
    pr,cr,sr=load_curve(args.rayleigh,args.sigma_default)
    pl,cl,sl=load_curve(args.love,args.sigma_default)
    h,vs,gamma,rho=load_model(args.model)
    result,m=invert_joint_bfgs(pr,cr,sr,pl,cl,sl,h,vs,gamma,rho,
        vpvs=args.vpvs, vs_bounds=(args.vs_min,args.vs_max),
        gamma_bounds=(args.gamma_min,args.gamma_max),
        lambda_gamma=args.lambda_gamma,lambda_smooth=args.lambda_smooth,
        lambda_vs=args.lambda_vs, maxiter=args.maxiter,
        fd_step=args.fd_step,scan_points=args.scan_points)
    keys=("thickness_km","vsv_km_s","vsh_km_s","gamma","xi","vp_km_s","rho_g_cm3")
    save_csv(args.output_dir/"inverted_model.csv", keys, [m[k] for k in keys])
    for wave,p,obs,pred in (("rayleigh",pr,cr,m["rayleigh_pred"]),
                            ("love",pl,cl,m["love_pred"])):
        save_csv(args.output_dir/f"{wave}_fit.csv",
                 ("period_s","observed_km_s","predicted_km_s","residual_km_s"),
                 (p,obs,pred,pred-obs))
    report={
        "success":bool(result.success), "message":str(result.message),
        "iterations":int(result.nit), "objective":float(result.fun),
        "rayleigh_rmse_km_s":float(np.sqrt(np.mean((m["rayleigh_pred"]-cr)**2))),
        "love_rmse_km_s":float(np.sqrt(np.mean((m["love_pred"]-cl)**2))),
        "model":"Restricted VTI, fixed thickness, rho, Vp/Vsv",
    }
    (args.output_dir/"summary.json").write_text(json.dumps(report,indent=2))
    if not args.no_plot:
        plot_results(args.output_dir,pr,cr,pl,cl,m)
    print(json.dumps(report,indent=2))
    print("Output:",args.output_dir.resolve())


if __name__=="__main__":
    main()
