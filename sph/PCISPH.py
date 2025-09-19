from BaseSPH import *
import numpy as np

@wp.kernel
def predict_velocity(
    v_pred: wp.array(dtype=wp.vec3), x_pred: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.vec3),
    a: wp.array(dtype=wp.vec3), a_p: wp.array(dtype=wp.vec3), dt: float
):
    tid = wp.tid()
    v_pred[tid] = v[tid] + (a[tid] + a_p[tid]) * dt
    x_pred[tid] = x[tid] + v_pred[tid] * dt
    
@wp.kernel
def solve_pressure(
    grid: wp.uint64, 
    x: wp.array(dtype=wp.vec3), x_pred: wp.array(dtype=wp.vec3),
    rho: wp.array(dtype=float), rho_pred: wp.array(dtype=float),
    rho_err: wp.array(dtype=float), p: wp.array(dtype=float),
    delta: float
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    neighbors = wp.hash_grid_query(grid, x[i], smoothing_radius)
    weight = float(0.0)
    for j in neighbors:
        weight += cubic_kernel(wp.length(x_pred[i] - x_pred[j]))
    rho1 = weight * mass
    den_err = rho1 - rho0
    pressure = delta * den_err
    if pressure < 0.0:
        pressure, den_err = 0.0, 0.0
    p[i] += pressure
    rho_pred[i] = rho1
    rho_err[i] = den_err
    
@wp.kernel
def compute_pressure_acceleration(
    grid: wp.uint64, 
    x: wp.array(dtype=wp.vec3),
    rho: wp.array(dtype=float),
    p: wp.array(dtype=float), a_p: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    neighbors = wp.hash_grid_query(grid, x[i], smoothing_radius)
    a_p[i] = wp.vec3(0.0)
    for j in neighbors:
        r_ij = x[i] - x[j]
        a_p[i] -= mass * (p[i] / rho[i]**2.0 + p[j] / rho[j]**2.0) * cubic_kernel_grad(r_ij)

class PCISPH(BaseSPH):
    def __init__(self):
        super().__init__()
        self.v_pred = None
        self.x_pred = None
        self.rho_pred = None
        self.p = None
        self.a_p = None
        self.rho_err = None
        self.beta = self.compute_beta()
        self.max_iter = 5
    
    def finalize(self):
        super().finalize()
        self.v_pred = wp.zeros(self.n, dtype=wp.vec3)
        self.x_pred = wp.zeros(self.n, dtype=wp.vec3)
        self.rho_pred = wp.zeros(self.n, dtype=float)
        self.p = wp.zeros(self.n, dtype=float)
        self.a_p = wp.zeros(self.n, dtype=wp.vec3)
        self.rho_err = wp.zeros(self.n, dtype=float)
    
    def compute_beta(self) -> float:
        bound = 1.5 * smoothing_radius
        points = uniform_lattice_generator(wp.vec3(-bound), wp.vec3(bound))
            
        denom1 = wp.vec3(0.0)
        denom2 = 0.0
        for p in points:
            r = wp.length(p)
            if r < smoothing_radius:
                grad = cubic_kernel_grad(-p)
                denom1 += grad
                denom2 += wp.length_sq(grad)
        denom = -wp.length_sq(denom1) - denom2
        return denom

    def compute_sim_dt(self):
        max_a = np.max(np.linalg.norm(self.a.numpy(), axis=1))
        # PCISPH can use a larger time step
        self.sim_dt = 5 * min(0.4 * smoothing_radius / 340, 0.25 * np.sqrt(smoothing_radius / max_a))
        print("sim_dt:", self.sim_dt)

    def compute_pressure_force(self):
        self.p.zero_()
        self.a_p.zero_()
        
        max_rho_err = 0.0
        for _ in range(self.max_iter):
            wp.launch(predict_velocity, dim=self.n,
                inputs=[self.v_pred, self.x_pred, self.v, self.x, self.a, self.a_p, self.sim_dt])
            wp.launch(resolve_collision, dim=self.n,
                inputs=[self.x_pred, self.v_pred, wp.array(self.collider, dtype=Collider)])
            # scale it to enforce more.
            delta = -2. / (2 * (mass * self.sim_dt / rho0)**2 * self.beta) \
                if abs(self.beta) > 0.0 else 0.0
            wp.launch(solve_pressure, dim=self.n,
                inputs=[self.grid.id, self.x, self.x_pred, self.rho, self.rho_pred, self.rho_err, self.p, delta])
            wp.launch(compute_pressure_acceleration, dim=self.n,
                inputs=[self.grid.id, self.x, self.rho_pred, self.p, self.a_p])
            max_rho_err = np.max(np.abs(self.rho_err.numpy()))
            if max_rho_err < rho0 * 0.01:
                break
        if max_rho_err >= rho0 * 0.01:
            print("max_rho_err:", max_rho_err)
            
        self.a += self.a_p