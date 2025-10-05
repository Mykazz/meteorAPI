from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import scipy as scp
import os
import json
import uvicorn

##########################################################################################################
# === PHYSICAL CONSTANTS ===
##########################################################################################################
G = 6.67430e-11
mass_Sun = 1.9891e30
mass_EM = 6.0457e24
mass_E = 5.97219e24
R_E = 6371000.0
R_E_DENSITY = 2700.0
G_ACCEL = 9.81
JOULES_PER_MEGATON = 4.184e15
AU = 149597870700.0

##########################################################################################################
# === ORBITAL MECHANICS ===
##########################################################################################################
def solve_Kepler_eq(M_rad, e, it=100, tol=1e-6):
    M = (M_rad + 2 * np.pi) % (2 * np.pi)
    E = M_rad + e * np.sin(M_rad) if e < 0.8 else np.pi
    for _ in range(it):
        f = (E - e * np.sin(E)) - M
        fp = (1 - e * np.cos(E))
        if abs(fp) < 1e-9:
            break
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    return np.mod(E, 2 * np.pi)


def heliocentric_to_ecliptic_rotational_transform(Omega_rad, I_rad, w_rad):
    Rz_Omega = np.array([
        [np.cos(Omega_rad), -np.sin(Omega_rad), 0],
        [np.sin(Omega_rad), np.cos(Omega_rad), 0],
        [0, 0, 1],
    ])
    Rx_I = np.array([
        [1, 0, 0],
        [0, np.cos(I_rad), -np.sin(I_rad)],
        [0, np.sin(I_rad), np.cos(I_rad)],
    ])
    Rz_w = np.array([
        [np.cos(w_rad), -np.sin(w_rad), 0],
        [np.sin(w_rad), np.cos(w_rad), 0],
        [0, 0, 1],
    ])
    return Rz_Omega @ Rx_I @ Rz_w


def compute_r_ecl_planet(T, a0, e0, I0_deg, L0_deg, w0_bar_deg, Omega0_deg,
                         da=0, de=0, dI=0, dL=0, dw_bar=0, dOmega=0):
    a = a0 + da * T
    e = e0 + de * T
    I = I0_deg + dI * T
    L = L0_deg + dL * T
    w_bar = w0_bar_deg + dw_bar * T
    Omega = Omega0_deg + dOmega * T
    w = w_bar - Omega
    M = L - w_bar
    I, w, Omega, M = map(np.deg2rad, [I, w, Omega, M])
    E = solve_Kepler_eq(M, e)
    Transform = heliocentric_to_ecliptic_rotational_transform(Omega, I, w)
    r_helio = np.array([a * (np.cos(E) - e), a * np.sqrt(1 - e**2) * np.sin(E), 0])
    return Transform @ r_helio


def compute_r_and_v_small_body(a, e, I_deg, w_deg, Omega_deg, M_deg):
    mu = G * mass_Sun
    I, w, Omega, M_rad = np.deg2rad(I_deg), np.deg2rad(w_deg), np.deg2rad(Omega_deg), np.deg2rad(M_deg)
    E = solve_Kepler_eq(M_rad, e)
    beta = e / (1 + np.sqrt(1 - e**2))
    nu = E + 2 * np.arctan2(beta * np.sin(E), (1 - beta * np.cos(E)))
    r = a * (1 - e * np.cos(E))
    r_perifocal = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    h = np.sqrt(mu * a * (1 - e**2))
    v_perifocal = np.array([-mu / h * np.sin(nu), mu / h * (e + np.cos(nu)), 0.0])
    Transform = heliocentric_to_ecliptic_rotational_transform(Omega, I, w)
    return Transform @ r_perifocal, Transform @ v_perifocal


def ast_accel(t, r_ast, a0_EM, e0, I0, L0, w0_bar, Omega0, da_EM, de, dI, dL, dw_bar, dOmega):
    r_Sun = np.array([0, 0, 0])
    r_EM = compute_r_ecl_planet(t, a0_EM, e0, I0, L0, w0_bar, Omega0, da_EM, de, dI, dL, dw_bar, dOmega)
    dist_to_sun = np.linalg.norm(r_ast - r_Sun)
    dist_to_EM = np.linalg.norm(r_ast - r_EM)
    return -G * (
        mass_Sun * (r_ast - r_Sun) / (dist_to_sun**3 + 1e-12)
        + mass_EM * (r_ast - r_EM) / (dist_to_EM**3 + 1e-12)
    )

##########################################################################################################
# === IMPACT PHYSICS ===
##########################################################################################################
def calculate_impact_radii(d_km, density, v, target_density=R_E_DENSITY, angle_deg=90):
    L = d_km * 1000.0
    rho_i = density
    rho_t = target_density
    theta = np.deg2rad(angle_deg)
    r_i = L / 2.0
    M = (4/3) * np.pi * (r_i**3) * rho_i
    E_k = 0.5 * M * v**2
    E_mt = E_k / JOULES_PER_MEGATON
    if E_mt < 1e-6:
        return {"Error": "Energy too low for meaningful calculation."}

    term_rho = (rho_t / rho_i)**(1/3)
    term_gv = (G_ACCEL * L) / (v**2 * np.sin(theta))
    term_E = E_k / (rho_t * G_ACCEL)
    D_c_transient = 1.16 * term_rho * (term_gv**0.22) * (term_E**0.28)
    D_c_final = D_c_transient * 1.5
    R_blast = 2.0 * (E_mt**(1/3))
    R_burn = 2.4 * (E_mt**(1/3))
    return {
        "Impactor Mass (kg)": M,
        "Kinetic Energy (J)": E_k,
        "TNT Equivalent (MT)": E_mt,
        "Transient Crater Diameter (km)": D_c_transient / 1000.0,
        "Final Crater Diameter (km)": D_c_final / 1000.0,
        "Blast Radius (km)": R_blast,
        "Thermal Burn Radius (km)": R_burn,
    }

##########################################################################################################
# === FASTAPI APP ===
##########################################################################################################
app = FastAPI(title="Asteroid Simulation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ allow Android / localhost / Railway access
    allow_methods=["*"],
    allow_headers=["*"],
)

##########################################################################################################
# === REQUEST MODEL ===
##########################################################################################################
class OrbitRequest(BaseModel):
    fullName: str
    spkid: str
    a: Optional[float] = 1.46
    e: Optional[float] = 0.223
    i: Optional[float] = 10.8
    node: Optional[float] = 304.0
    peri: Optional[float] = 179.0
    M: Optional[float] = 311.0


##########################################################################################################
# === MAIN SIMULATION ENDPOINT ===
##########################################################################################################
@app.post("/v1/simulate")
def simulate(req: OrbitRequest):
    try:
        # --- Earth orbit ---
        a0_EM = 1.00000018 * AU
        da_EM = -0.00000003 * AU / (3.1556926e9)
        e0 = 0.01673163
        de = -0.00003661 / (3.1556926e9)
        I0 = -0.00001531
        dI = -0.01294668 / (3.1556926e9)
        L0 = 100.46457166
        dL = 35999.37244981 / (3.1556926e9)
        w0_bar = 102.93005885
        dw_bar = 0.31795260 / (3.1556926e9)
        Omega0 = -5.11260389
        dOmega = -0.24123856 / (3.1556926e9)

        # --- Earth trajectory ---
        T = np.linspace(0, 5 * 365.25 * 86400, 6000)
        r_EM = np.array([
            compute_r_ecl_planet(t, a0_EM, e0, I0, L0, w0_bar, Omega0,
                                 da_EM, de, dI, dL, dw_bar, dOmega)
            for t in T
        ])

        # --- Asteroid parameters ---
        a_ast, e_ast, i_ast, w_ast, node_ast, M_ast = (
            req.a * AU,
            req.e,
            req.i,
            req.peri,
            req.node,
            req.M,
        )

        r_ast0, v_ast0 = compute_r_and_v_small_body(a_ast, e_ast, i_ast, w_ast, node_ast, M_ast)
        y0 = np.hstack([r_ast0, v_ast0])

        # --- Differential Equation Integration ---
        def deriv(t, y):
            r = y[:3]
            a_vec = ast_accel(t, r, a0_EM, e0, I0, L0, w0_bar, Omega0,
                              da_EM, de, dI, dL, dw_bar, dOmega)
            return np.concatenate([y[3:], a_vec])

        sol = scp.integrate.solve_ivp(deriv, (T[0], T[-1]), y0, t_eval=T, rtol=1e-9, atol=1e-12)
        v_mean = np.mean(np.linalg.norm(sol.y[3:], axis=0))

        # --- Impact Physics ---
        crater = calculate_impact_radii(1.0, 3000, v_mean)
        if "Error" in crater:
            crater = {"Final Crater Diameter (km)": 0, "Blast Radius (km)": 0, "Thermal Burn Radius (km)": 0}

        data = {
            "x_ast": sol.y[0, :].tolist(),
            "y_ast": sol.y[1, :].tolist(),
            "z_ast": sol.y[2, :].tolist(),
            "x_earth": r_EM[:, 0].tolist(),
            "y_earth": r_EM[:, 1].tolist(),
            "z_earth": r_EM[:, 2].tolist(),
            "impact": {
                "lat": 55.1694,
                "lng": 23.8813,
                "radius_km": crater["Blast Radius (km)"],
            },
        }

        # --- Write JSON for visualization ---
        os.makedirs("public/assets/r3f_demo/assets/data", exist_ok=True)
        with open("public/assets/r3f_demo/assets/data/asteroid_data.json", "w") as f:
            json.dump(data, f, indent=2)

        return {"ok": True, "input": req.dict(), "data": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


##########################################################################################################
# === ENTRY POINT ===
##########################################################################################################
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Railway sets PORT automatically
    uvicorn.run("main:app", host="0.0.0.0", port=port)
