import warp as wp
import numpy as np

dim = 3
p_radius = 0.01
p_diameter = p_radius * 2.0
smoothing_radius = p_radius * 4.0
rho0 = 1000.0
viscosity = 0.002
surface_tension = 0.01
lower_boundary = wp.vec3(0.0, 0.0, 0.0)
upper_boundary = wp.vec3(1.0, 2.0, 2.0)

@wp.func
def cubic_kernel(r: float) -> float:
    q = r / smoothing_radius
    sigma = 8.0 / (wp.pi * smoothing_radius**3.0)
    if q <= 0.5:
        return sigma * (6.0 * (q**3.0 - q**2.0) + 1.0)
    elif q <= 1.0:
        return sigma * (2.0 * (1.0 - q)**3.0)
    else:
        return 0.0

@wp.func
def cubic_kernel_grad(r: wp.vec3) -> wp.vec3:
    r_len = wp.length(r)
    q = r_len / smoothing_radius
    sigma = 8.0 / (wp.pi * smoothing_radius**3.0)
    if r_len > 1e-5:
        grad_r = r / (r_len * smoothing_radius)
        if q <= 0.5:
            return sigma * q * (18.0 * q - 12.0) * grad_r
        elif q <= 1.0:
            return -sigma * 6.0 * (1.0 - q)**2.0 * grad_r
        else:
            return wp.vec3()
    else:
        return wp.vec3()

def uniform_lattice_generator(pos_start: wp.vec3, pos_end: wp.vec3):
    # BCC lattice - not applicable
    # points = []
    # spacing = p_radius
    # k = 0
    # while k * spacing + pos_start.z <= pos_end.z:
    #     z = k * spacing + pos_start.z
    #     offset_x = p_radius if k % 2 == 1 else 0.0
    #     j = 0
    #     while j * spacing + pos_start.y <= pos_end.y:
    #         y = j * spacing + pos_start.y
    #         offset_y = p_radius if j % 2 == 1 else 0.0
    #         i = 0
    #         while i * spacing + pos_start.x <= pos_end.x:
    #             x = i * spacing + pos_start.x
    #             points.append(wp.vec3(x + offset_x, y + offset_y, z))
    #             i += 1
    #         j += 1
    #     k += 1
    x = np.arange(pos_start.x + p_radius, pos_end.x, p_diameter)
    y = np.arange(pos_start.y + p_radius, pos_end.y, p_diameter)
    z = np.arange(pos_start.z + p_radius, pos_end.z, p_diameter)
    x, y, z = np.meshgrid(x, y, z)
    points = [wp.vec3(px, py, pz) for px, py, pz in zip(x.flatten(), y.flatten(), z.flatten())]
    return points

def compute_mass() -> float:
    bound = 1.0 * smoothing_radius
    points = uniform_lattice_generator(wp.vec3(-bound), wp.vec3(bound))
    max_density = 0.0
    for p in points:
        max_density += cubic_kernel(wp.length(p))
    return rho0 / max_density if max_density > 0.0 else 0.0

mass = compute_mass()
print(mass)
    
@wp.kernel
def add_emitter(x: wp.array(dtype=wp.vec3), n: int, origin: wp.vec3, dim_x: int, dim_y: int, dim_z: int):
    i, j, k = wp.tid()
    tid = i * dim_y * dim_z + j * dim_z + k
    x[n + tid] = origin + wp.vec3(
        float(i) * p_diameter + p_radius,
        float(j) * p_diameter + p_radius,
        float(k) * p_diameter + p_radius
    )
    
@wp.kernel
def compute_density(
    grid: wp.uint64, x: wp.array(dtype=wp.vec3), rho: wp.array(dtype=float)
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    neighbors = wp.hash_grid_query(grid, x[i], smoothing_radius)
    rho[i] = 0.0
    for j in neighbors:
        rho[i] += cubic_kernel(wp.length(x[i] - x[j]))
    rho[i] *= mass

@wp.kernel
def compute_non_pressure_forces(
    grid: wp.uint64, x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
    rho: wp.array(dtype=float), a: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    a[i] = wp.vec3(0.0, -9.81, 0.0)
    neighbors = wp.hash_grid_query(grid, x[i], smoothing_radius)
    for j in neighbors:
        r_ij = x[i] - x[j]
        # viscosity
        r_len_sq = wp.length_sq(r_ij)
        a[i] += 2.0 * (float(dim) + 2.0) * viscosity * (mass / rho[j]) * wp.dot(v[i] - v[j], r_ij) / (
            r_len_sq + 0.01 * smoothing_radius**2.0) * cubic_kernel_grad(r_ij)
        # surface tension
        if r_len_sq > p_diameter * p_diameter:
            a[i] -= surface_tension * r_ij * cubic_kernel(wp.length(r_ij))
        else:
            a[i] -= surface_tension * r_ij * cubic_kernel(p_diameter)

@wp.kernel
def advect(
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), a: wp.array(dtype=wp.vec3), dt: float
):
    tid = wp.tid()
    v[tid] += a[tid] * dt
    x[tid] += v[tid] * dt

@wp.struct
class Collider:
    # spherical collider
    origin: wp.vec3
    radius: float

@wp.kernel
def resolve_collision(
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
    collider: wp.array(dtype=Collider)
):
    tid = wp.tid()
    decay_factor = 0.95
    
    # boundary
    for k in range(3):
        if x[tid][k] < lower_boundary[k] + p_radius:
            x[tid][k] = lower_boundary[k] + p_radius
            if v[tid][k] < 0:
                v[tid][k] *= -decay_factor
        if x[tid][k] > upper_boundary[k] - p_radius:
            x[tid][k] = upper_boundary[k] - p_radius
            if v[tid][k] > 0:
                v[tid][k] *= -decay_factor
            
    # spherical collider
    for _ in range(collider.shape[0]):
        c_origin = collider[_].origin
        c_radius = collider[_].radius
        dir = x[tid] - c_origin
        dist = wp.length(dir)
        if dist < c_radius + p_radius:
            n = dir / dist
            x[tid] = c_origin + n * (c_radius + p_radius)
            v_normal = wp.dot(v[tid], n) * n
            v_tangent = v[tid] - v_normal
            v[tid] = v_tangent - v_normal * decay_factor

class BaseSPH:
    def __init__(self):
        self.dt = 1 / 60
        self.sim_dt = 0.01 * smoothing_radius
        self.n = 0
        self.x = None
        self.v = None
        self.a = None
        self.rho = None
        self.bottom_left = wp.vec3(100)
        self.upper_right = wp.vec3(-100)
        self.grid = None
        self.collider = []
        self.cpu_x = []
        
    def add_cube_emitter(self, pos_start: wp.vec3, pos_end: wp.vec3):
        points = uniform_lattice_generator(pos_start, pos_end)
        num_new_particles = len(points)
        print(f"{num_new_particles} new particles generated.")
        
        self.cpu_x.extend(points)
        self.n += num_new_particles
        self.bottom_left = wp.min(self.bottom_left, pos_start)
        self.upper_right = wp.max(self.upper_right, pos_end)
        
    def add_collider(self, origin: wp.vec3, radius: float):
        collider = Collider()
        collider.origin = origin
        collider.radius = radius
        self.collider.append(collider)
    
    def set_grid(self):
        size = (self.upper_right - self.bottom_left) / smoothing_radius
        self.grid = wp.HashGrid(int(size.x), int(size.y), int(size.z))
    
    def finalize(self):
        self.x = wp.from_numpy(np.array(self.cpu_x, dtype=np.float32).flatten(), dtype=wp.vec3)
        self.v = wp.zeros(self.n, dtype=wp.vec3)
        self.a = wp.zeros(self.n, dtype=wp.vec3)
        self.rho = wp.zeros(self.n, dtype=float)
        self.set_grid()
        
    def precompute(self):
        pass
        
    def compute_pressure_force(self):
        pass
        
    def compute_sim_dt(self):
        max_a = np.max(np.linalg.norm(self.a.numpy(), axis=1))
        self.sim_dt = min(0.4 * smoothing_radius / 100, np.sqrt(smoothing_radius / max_a))

    def step(self, verbose=False):
        with wp.ScopedTimer("build_grid", active=verbose):
            self.grid.build(self.x, smoothing_radius)
        with wp.ScopedTimer("compute_density", active=verbose):
            wp.launch(compute_density, dim=self.n,
                inputs=[self.grid.id, self.x, self.rho])
        self.precompute()
        with wp.ScopedTimer("compute_non_pressure_forces", active=verbose):
            wp.launch(compute_non_pressure_forces, dim=self.n,
                inputs=[self.grid.id, self.x, self.v, self.rho, self.a])
        with wp.ScopedTimer("compute_pressure_force", active=verbose):
            self.compute_pressure_force()
        with wp.ScopedTimer("advect", active=verbose):
            wp.launch(advect, dim=self.n,
                inputs=[self.x, self.v, self.a, self.sim_dt])
        with wp.ScopedTimer("resolve_collision", active=verbose):
            wp.launch(resolve_collision, dim=self.n,
                inputs=[self.x, self.v, wp.array(self.collider, dtype=Collider)])