

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# -------------------- PARAMETERS --------------------
# domain (rectangle)
region = ((0.0, 10.0), (0.0, 6.0))  # (xmin,xmax),(ymin,ymax)
width = region[0][1] - region[0][0]
height = region[1][1] - region[1][0]
area = width * height

# PPP parameters: either set fixed_lambda or target phi (coverage)
use_target_phi = True
phi_target = 0.2       # if use_target_phi True, compute lambda from phi
fixed_lambda = None     # set to a number to override phi

# bubble radius options
fixed_radius = 0.18
random_radii = False    # if True, sample radii from radius_dist
radius_dist = {"type":"normal", "mean":0.18, "std":0.03, "min":0.05, "max":0.4}

# refractive indices
n_water = 1.3330
n_bubble = 1.0003

# grid for n(x,y)
Nx, Ny = 500, 300   # grid resolution; increase for more accuracy
xg = np.linspace(region[0][0], region[0][1], Nx)
yg = np.linspace(region[1][0], region[1][1], Ny)
dx = xg[1] - xg[0]
dy = yg[1] - yg[0]
XX, YY = np.meshgrid(xg, yg, indexing='xy')  # shape (Ny, Nx)

# smoothing / edge choices
use_soft_edges = True      # True: smooth transition near disk boundary; False: sharp boolean mask
soft_sigma_frac = 0.25     # sigma = r * soft_sigma_frac

# ray tracing parameters
n_rays = 11
y_span = (region[1][0] + 0.2, region[1][1] - 0.2)
angle_deg = 0.0    # initial angle (0 = left->right)
ds = 0.005         # ODE step (arc-length parameter)
max_steps = 15000

# padding: simulate PPP in slightly larger region so bubbles outside can cover interior near edges
pad = max(1.0, fixed_radius * 3.0)
sim_region = ((region[0][0]-pad, region[0][1]+pad), (region[1][0]-pad, region[1][1]+pad))

# RNG seed for reproducibility
np.random.seed(3)


# -------------------- helper functions --------------------
def lambda_from_phi(phi, r_sq_mean):
    """Return homogeneous PPP intensity λ for target coverage phi and E[R^2]."""
    return -math.log(1 - phi) / (math.pi * r_sq_mean)

def sample_radii(N):
    """Return array of radii of length N."""
    if not random_radii:
        return np.full(N, fixed_radius)
    typ = radius_dist.get("type", "normal")
    if typ == "normal":
        rs = np.random.normal(radius_dist["mean"], radius_dist["std"], size=N)
    elif typ == "uniform":
        low = radius_dist["low"]; high = radius_dist["high"]
        rs = np.random.uniform(low, high, size=N)
    else:
        rs = np.full(N, fixed_radius)
    rs = np.clip(rs, radius_dist.get("min", 1e-6), radius_dist.get("max", 10.0))
    return rs


# -------------------- simulate PPP centers (homogeneous) --------------------
if fixed_lambda is not None:
    lam = float(fixed_lambda)
else:
    if use_target_phi:
        if random_radii:
            # quick Monte Carlo estimate of E[R^2]
            trial = np.random.normal(radius_dist["mean"], radius_dist["std"], size=200000)
            trial = np.clip(trial, radius_dist.get("min", 1e-6), radius_dist.get("max", 10.0))
            E_R2 = np.mean(trial**2)
        else:
            E_R2 = fixed_radius**2
        lam = lambda_from_phi(phi_target, E_R2)
    else:
        lam = 1.0

sim_x0, sim_x1 = sim_region[0]
sim_y0, sim_y1 = sim_region[1]
sim_area = (sim_x1 - sim_x0) * (sim_y1 - sim_y0)

# draw Poisson(N) in padded simulation region
N_sim = np.random.poisson(lam * sim_area)
xs_all = np.random.uniform(sim_x0, sim_x1, N_sim)
ys_all = np.random.uniform(sim_y0, sim_y1, N_sim)
rs_all = sample_radii(N_sim)

# keep only bubbles that could affect the main window (to save work)
keep_mask = (xs_all >= region[0][0] - rs_all) & (xs_all <= region[0][1] + rs_all) & \
            (ys_all >= region[1][0] - rs_all) & (ys_all <= region[1][1] + rs_all)
xs = xs_all[keep_mask]; ys = ys_all[keep_mask]; rs = rs_all[keep_mask]
N = xs.size

print(f"Simulated {N_sim} centers in padded region (λ={lam:.4f}). {N} centers affect window after cropping.")


# -------------------- build refractive-index field n(x,y) --------------------
n_field = np.full(XX.shape, n_water, dtype=float)

if use_soft_edges:
    # soft gaussian-like mask around boundary
    for cx, cy, r in zip(xs, ys, rs):
        sigma = max(1e-8, r * soft_sigma_frac)
        d = np.sqrt((XX - cx)**2 + (YY - cy)**2)
        mask = np.exp(-0.5 * ((d - r) / sigma)**2)
        mask *= (d <= (r + 3*sigma))
        mask[d <= r] = 1.0
        #Forces overlapping bubbles to combine
        n_field = mask * n_bubble + (1 - mask) * n_field
else:
    # sharp boolean mask: set to bubble index inside any disk
    for cx, cy, r in zip(xs, ys, rs):
        mask = (XX - cx)**2 + (YY - cy)**2 <= r*r
        n_field[mask] = n_bubble

# compute gradients (finite differences)
# np.gradient expects first axis = y, second = x when passing yg, xg spacings
dn_dy, dn_dx = np.gradient(n_field, yg, xg, edge_order=2)


# -------------------- empirical covered fraction (grid estimate) --------------------
covered_mask = n_field < (n_water - 1e-8)
empirical_phi = float(np.mean(covered_mask))
print(f"Empirical covered fraction on grid: {empirical_phi:.4f} (target φ = {phi_target})")


# -------------------- interpolation helpers --------------------
def bilinear_interpolate(grid_x, grid_y, field, x, y):
    """Bilinear interpolate 'field' defined on grid_x x grid_y at point (x,y)."""
    nx = grid_x.size; ny = grid_y.size
    # clamp indices
    if x <= grid_x[0]:
        ix = 0
    elif x >= grid_x[-1]:
        ix = nx - 2
    else:
        ix = np.searchsorted(grid_x, x) - 1
    if y <= grid_y[0]:
        iy = 0
    elif y >= grid_y[-1]:
        iy = ny - 2
    else:
        iy = np.searchsorted(grid_y, y) - 1
    ix = np.clip(ix, 0, nx - 2)
    iy = np.clip(iy, 0, ny - 2)
    x1, x2 = grid_x[ix], grid_x[ix + 1]
    y1, y2 = grid_y[iy], grid_y[iy + 1]
    Q11 = field[iy, ix]; Q21 = field[iy, ix + 1]; Q12 = field[iy + 1, ix]; Q22 = field[iy + 1, ix + 1]
    if (x2 - x1) == 0 or (y2 - y1) == 0:
        return Q11
    tx = (x - x1) / (x2 - x1)
    ty = (y - y1) / (y2 - y1)
    return (Q11 * (1 - tx) * (1 - ty)
            + Q21 * tx * (1 - ty)
            + Q12 * (1 - tx) * ty
            + Q22 * tx * ty)

def interp_n(x, y):
    return bilinear_interpolate(xg, yg, n_field, x, y)

def interp_grad(x, y):
    gx = bilinear_interpolate(xg, yg, dn_dx, x, y)
    gy = bilinear_interpolate(xg, yg, dn_dy, x, y)
    return np.array([gx, gy], dtype=float)


# -------------------- ray ODE and integrator --------------------
def ray_rhs(state):
    # state: [x, y, vx, vy]
    x, y, vx, vy = state
    v = np.array([vx, vy], dtype=float)
    nval = interp_n(x, y)
    gradn = interp_grad(x, y)
    vdotgrad = float(np.dot(v, gradn))
    denom = nval if abs(nval) > 1e-12 else 1.0
    dvds = (gradn - v * vdotgrad) / denom
    return np.array([vx, vy, dvds[0], dvds[1]], dtype=float)

def rk4_step(state, h):
    k1 = ray_rhs(state)
    k2 = ray_rhs(state + 0.5 * h * k1)
    k3 = ray_rhs(state + 0.5 * h * k2)
    k4 = ray_rhs(state + h * k3)
    return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# -------------------- trace rays --------------------
y_initials = np.linspace(y_span[0], y_span[1], n_rays)
theta = math.radians(angle_deg)
v0 = np.array([math.cos(theta), math.sin(theta)], dtype=float)
x_start = region[0][0] - 0.2

rays = []
for y0 in y_initials:
    state = np.array([x_start, y0, v0[0], v0[1]], dtype=float)
    traj = [state.copy()]
    for step in range(max_steps):
        state = rk4_step(state, ds)
        # stop if outside domain (with margin)
        if state[0] > region[0][1] + 0.5 or state[0] < region[0][0] - 1.0 or \
           state[1] < region[1][0] - 1.0 or state[1] > region[1][1] + 1.0:
            traj.append(state.copy())
            break
        traj.append(state.copy())
    rays.append(np.array(traj))


# -------------------- PLOTTING --------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(region[0]); ax.set_ylim(region[1])
ax.set_aspect('equal')
ax.set_title("Rays through bubbly water — n(x,y) and trajectories")

# show n_field as image
im = ax.imshow(n_field, origin='lower',
               extent=(region[0][0], region[0][1], region[1][0], region[1][1]),
               alpha=0.95)
plt.colorbar(im, ax=ax, label='n(x,y)')

# overlay bubble boundaries (thin)
for cx, cy, r in zip(xs, ys, rs):
    circ = Circle((cx, cy), r, fill=False, linewidth=0.3, edgecolor='k', alpha=0.25)
    ax.add_patch(circ)

# plot rays
for traj in rays:
    ax.plot(traj[:, 0], traj[:, 1], linewidth=1.0)

ax.set_xlabel("x"); ax.set_ylabel("y")
plt.savefig('./ray_tracing.png')
plt.show()
