import warp as wp
import warp.sparse as sparse
import warp.optim.linear as linear

grid_width = 256
grid_height = 256
res = (grid_width, grid_height)
res_flat = grid_width * grid_height
max_nnz = res_flat * 5
res_pad = (grid_width + 1, grid_height + 1)
buoyancy_den_factor = -9.81
buoyancy_temp_factor = 1.0
den_decay_factor = 1.0 / 1000.0
temp_decay_factor = 1.0 / 1000.0
ambient_temp = 0.0
gravity = -9.81 # not used
eps = 1e-8

@wp.func
def scalar_at(f: wp.array2d(dtype=float), i: int, j: int) -> float:
    i = wp.clamp(i, 0, grid_width - 1)
    j = wp.clamp(j, 0, grid_height - 1)
    return f[i, j]

@wp.func
def cerp(a: float, b: float, c: float, d: float, t: float) -> float:
    w_1 = -t * (t - 1.) * (t - 2.) / 6.
    w0 = (t + 1.) * (t - 1.) * (t - 2.) / 2.
    w1 = -t * (t + 1.) * (t - 2.) / 2.
    w2 = t * (t + 1.) * (t - 1.) / 6.
    return w_1 * a + w0 * b + w1 * c + w2 * d

@wp.func
def scalar_bilinear(f: wp.array2d(dtype=float), x: float, y: float) -> float:
    lx, ly = int(x - 0.5), int(y - 0.5)
    tx, ty = x - 0.5 - float(lx), y - 0.5 - float(ly)
    
    s0 = wp.lerp(scalar_at(f, lx, ly), scalar_at(f, lx + 1, ly), tx)
    s1 = wp.lerp(scalar_at(f, lx, ly + 1), scalar_at(f, lx + 1, ly + 1), tx)
    s = wp.lerp(s0, s1, ty)
    return s

@wp.func
def vel_lin(u: wp.array2d(dtype=wp.vec2), x: float, y: float) -> wp.vec2:
    lx, ly = int(wp.floor(x)), int(wp.floor(y - 0.5))
    tx, ty = x - float(lx), y - 0.5 - float(ly)
    v0 = wp.lerp(vel_at(u, lx, ly).x, vel_at(u, lx + 1, ly).x, tx)
    v1 = wp.lerp(vel_at(u, lx, ly + 1).x, vel_at(u, lx + 1, ly + 1).x, tx)
    uu = wp.lerp(v0, v1, ty)
    lx, ly = int(wp.floor(x - 0.5)), int(wp.floor(y))
    tx, ty = x - 0.5 - float(lx), y - float(ly)
    v0 = wp.lerp(vel_at(u, lx, ly).y, vel_at(u, lx + 1, ly).y, tx)
    v1 = wp.lerp(vel_at(u, lx, ly + 1).y, vel_at(u, lx + 1, ly + 1).y, tx)
    vv = wp.lerp(v0, v1, ty)
    return wp.vec2(uu, vv)

@wp.func
def scalar_bicubic(f: wp.array2d(dtype=float), x: float, y: float) -> float:
    lx, ly = int(wp.floor(x - 0.5)), int(wp.floor(y - 0.5))
    tx, ty = x - 0.5 - float(lx), y - 0.5 - float(ly)

    c0 = cerp(scalar_at(f, lx - 1, ly - 1), scalar_at(f, lx, ly - 1), scalar_at(f, lx + 1, ly - 1), scalar_at(f, lx + 2, ly - 1), tx)
    c1 = cerp(scalar_at(f, lx - 1, ly), scalar_at(f, lx, ly), scalar_at(f, lx + 1, ly), scalar_at(f, lx + 2, ly), tx)
    c2 = cerp(scalar_at(f, lx - 1, ly + 1), scalar_at(f, lx, ly + 1), scalar_at(f, lx + 1, ly + 1), scalar_at(f, lx + 2, ly + 1), tx)
    c3 = cerp(scalar_at(f, lx - 1, ly + 2), scalar_at(f, lx, ly + 2), scalar_at(f, lx + 1, ly + 2), scalar_at(f, lx + 2, ly + 2), tx)
    return cerp(c0, c1, c2, c3, ty)

@wp.func
def vel_at(u: wp.array2d(dtype=wp.vec2), i: int, j: int) -> wp.vec2:
    if i > grid_width or i < 0 or j > grid_height or j < 0:
        return wp.vec2()
    return u[i, j]

@wp.func
def vel_bicubic(u: wp.array2d(dtype=wp.vec2), x: float, y: float) -> wp.vec2:
    lx, ly = int(wp.floor(x)), int(wp.floor(y - 0.5))
    tx, ty = x - float(lx), y - 0.5 - float(ly)
    v_1_1, v0_1, v1_1, v2_1 = vel_at(u, lx - 1, ly - 1), vel_at(u, lx, ly - 1), vel_at(u, lx + 1, ly - 1), vel_at(u, lx + 2, ly - 1)
    v_10, v00, v10, v20 = vel_at(u, lx - 1, ly), vel_at(u, lx, ly), vel_at(u, lx + 1, ly), vel_at(u, lx + 2, ly)
    v_11, v01, v11, v21 = vel_at(u, lx - 1, ly + 1), vel_at(u, lx, ly + 1), vel_at(u, lx + 1, ly + 1), vel_at(u, lx + 2, ly + 1)
    v_12, v02, v12, v22 = vel_at(u, lx - 1, ly + 2), vel_at(u, lx, ly + 2), vel_at(u, lx + 1, ly + 2), vel_at(u, lx + 2, ly + 2)
    u0 = cerp(v_1_1.x, v0_1.x, v1_1.x, v2_1.x, tx)
    u1 = cerp(v_10.x, v00.x, v10.x, v20.x, tx)
    u2 = cerp(v_11.x, v01.x, v11.x, v21.x, tx)
    u3 = cerp(v_12.x, v02.x, v12.x, v22.x, tx)
    uu = cerp(u0, u1, u2, u3, ty)
    lx, ly = int(wp.floor(x - 0.5)), int(wp.floor(y))
    tx, ty = x - 0.5 - float(lx), y - float(ly)
    v_1_1, v0_1, v1_1, v2_1 = vel_at(u, lx - 1, ly - 1), vel_at(u, lx, ly - 1), vel_at(u, lx + 1, ly - 1), vel_at(u, lx + 2, ly - 1)
    v_10, v00, v10, v20 = vel_at(u, lx - 1, ly), vel_at(u, lx, ly), vel_at(u, lx + 1, ly), vel_at(u, lx + 2, ly)
    v_11, v01, v11, v21 = vel_at(u, lx - 1, ly + 1), vel_at(u, lx, ly + 1), vel_at(u, lx + 1, ly + 1), vel_at(u, lx + 2, ly + 1)
    v_12, v02, v12, v22 = vel_at(u, lx - 1, ly + 2), vel_at(u, lx, ly + 2), vel_at(u, lx + 1, ly + 2), vel_at(u, lx + 2, ly + 2)
    v0 = cerp(v_1_1.y, v0_1.y, v1_1.y, v2_1.y, tx)
    v1 = cerp(v_10.y, v00.y, v10.y, v20.y, tx)
    v2 = cerp(v_11.y, v01.y, v11.y, v21.y, tx)
    v3 = cerp(v_12.y, v02.y, v12.y, v22.y, tx)
    vv = cerp(v0, v1, v2, v3, ty)
    return wp.vec2(uu, vv)

@wp.kernel
def update_emitter(rho: wp.array2d(dtype=float), temp: wp.array2d(dtype=float), u: wp.array2d(dtype=wp.vec2), radius: float, vel: wp.vec2):
    i, j = wp.tid()
    ii, jj = float(i), float(j)
    w, h = float(grid_width), float(grid_height)
    d = wp.length(wp.vec2(ii + 0.5 - w / 2., jj + 0.5 - h / 2.)) # center
    d_u = wp.length(wp.vec2(ii - w / 2., jj + 0.5 - h / 2.))
    d_v = wp.length(wp.vec2(ii + 0.5 - w / 2., jj - h / 2.))

    if d < radius and i < grid_width and j < grid_height:
        rho[i, j] = 2.508
        temp[i, j] = 100.0
        if d_u < radius and j < grid_height:
            u[i, j].x = vel.x
        if d_v < radius and i < grid_width:
            u[i, j].y = vel.y

@wp.kernel
def apply_buoyancy(rho: wp.array2d(dtype=float), temp: wp.array2d(dtype=float), u: wp.array2d(dtype=wp.vec2), dt: float):
    i, j = wp.tid()
    # u is zero
    if i < grid_width and scalar_at(rho, i, j) + scalar_at(rho, i, j - 1) > eps:
        buoyancy_v = buoyancy_den_factor * (scalar_at(rho, i, j) + scalar_at(rho, i, j - 1)) * 0.5 +\
            buoyancy_temp_factor * ((scalar_at(temp, i, j) + scalar_at(temp, i, j - 1)) * 0.5 - ambient_temp)
        u[i, j].y += dt * (scalar_at(rho, i, j) + scalar_at(rho, i, j - 1)) * 0.5 * buoyancy_v

@wp.kernel
def divergence(rho: wp.array2d(dtype=float), u: wp.array2d(dtype=wp.vec2), div: wp.array2d(dtype=float)):
    i, j = wp.tid()
    if rho[i, j] < eps:
        div[i, j] = 0.0
        return
    du_dx = u[i + 1, j].x - u[i, j].x
    dv_dy = u[i, j + 1].y - u[i, j].y
    div[i, j] = du_dx + dv_dy
    
@wp.kernel
def substract_pressure(rho: wp.array2d(dtype=float), u: wp.array2d(dtype=wp.vec2), p: wp.array2d(dtype=float)):
    i, j = wp.tid()
    if rho[i, j] > eps or scalar_at(rho, i - 1, j) > eps:
        u[i, j].x -= p[i, j] - scalar_at(p, i - 1, j)
    if rho[i, j] > eps or scalar_at(rho, i, j - 1) > eps:
        u[i, j].y -= p[i, j] - scalar_at(p, i, j - 1)

@wp.func
def backtrace(u: wp.array2d(dtype=wp.vec2), p: wp.vec2, dt: float) -> wp.vec2:
    # RK3
    k1 = vel_bicubic(u, p.x, p.y)
    p1 = p - 0.5 * dt * k1
    k2 = vel_bicubic(u, p1.x, p1.y)
    p2 = p - 0.75 * dt * k2
    k3 = vel_bicubic(u, p2.x, p2.y)
    return p - dt * (2./9. * k1 + 3./9. * k2 + 4./9. * k3)
    # return p - dt * vel_bicubic(u, p.x, p.y)

@wp.kernel
def advect(
        u0: wp.array2d(dtype=wp.vec2), u1: wp.array2d(dtype=wp.vec2),
        rho0: wp.array2d(dtype=float), rho1: wp.array2d(dtype=float),
        temp0: wp.array2d(dtype=float), temp1: wp.array2d(dtype=float),
        dt: float
    ):
    i, j = wp.tid()
    if i < grid_width and j < grid_height:
        p = backtrace(u0, wp.vec2(float(i) + 0.5, float(j) + 0.5), dt)
        rho1[i, j] = wp.max(0., scalar_bicubic(rho0, p.x, p.y) * (1.0 - den_decay_factor))
        temp1[i, j] = wp.max(ambient_temp, scalar_bicubic(temp0, p.x, p.y) * (1.0 - temp_decay_factor))
        # rho1[i, j] = wp.max(0., scalar_bilinear(rho0, p.x, p.y) * (1.0 - den_decay_factor))
        # temp1[i, j] = wp.max(ambient_temp, scalar_bilinear(temp0, p.x, p.y) * (1.0 - temp_decay_factor))
    if j < grid_height:
        u_p = backtrace(u0, wp.vec2(float(i), float(j) + 0.5), dt)
        u1[i, j].x = vel_bicubic(u0, u_p.x, u_p.y).x
        # u1[i, j].x = vel_lin(u0, u_p.x, u_p.y).x
    if i < grid_width:
        v_p = backtrace(u0, wp.vec2(float(i) + 0.5, float(j)), dt)
        u1[i, j].y = vel_bicubic(u0, v_p.x, v_p.y).y
        # u1[i, j].x = vel_lin(u0, u_p.x, u_p.y).x

@wp.kernel
def fill_indices(rho: wp.array2d(dtype=float), row: wp.array(dtype=int), col: wp.array(dtype=int), val: wp.array(dtype=float)):
    i, j = wp.tid()
    if rho[i, j] < eps:
        return
    center = i * grid_height + j
    base = center * 5
    offset = 0
    if i > 0:
        if rho[i - 1, j] > eps:
            row[base + offset] = center
            col[base + offset] = center - grid_height
            val[base + offset] = 1.0
    offset += 1
    if i < grid_width - 1:
        if rho[i + 1, j] > eps:
            row[base + offset] = center
            col[base + offset] = center + grid_height
            val[base + offset] = 1.0
    offset += 1
    if j > 0:
        if rho[i, j - 1] > eps:
            row[base + offset] = center
            col[base + offset] = center - 1
            val[base + offset] = 1.0
    offset += 1
    if j < grid_height - 1:
        if rho[i, j + 1] > eps:
            row[base + offset] = center
            col[base + offset] = center + 1
            val[base + offset] = 1.0
        # up lid is closed
        offset += 1
    row[base + offset] = center
    col[base + offset] = center
    val[base + offset] = float(-offset)

class Smoke:
    def __init__(self):
        self.frame_dt = 1.0 / 60.0
        self.substeps = 2
        self.sim_dt = self.frame_dt / self.substeps
        self.elapsed_time = 0.0
        # MAC grid of velocity
        self.u0 = wp.zeros(res_pad, dtype=wp.vec2)
        self.u1 = wp.zeros(res_pad, dtype=wp.vec2)
        # density
        self.rho0 = wp.zeros(res, dtype=float)
        self.rho1 = wp.zeros(res, dtype=float)
        # temperature
        self.temp0 = wp.zeros(res, dtype=float)
        self.temp1 = wp.zeros(res, dtype=float)
        # pressure
        self.p = wp.zeros(res, dtype=float)
        # divergence
        self.u_div = wp.zeros(res, dtype=float)
        # assemble laplacian
        self.row_idx = wp.zeros(max_nnz, dtype=int)
        self.col_idx = wp.zeros(max_nnz, dtype=int)
        self.val = wp.zeros(max_nnz, dtype=float)

    def pressure_project(self):
        wp.launch(divergence, dim=res, inputs=[self.rho0, self.u0, self.u_div])

        self.row_idx.zero_()
        self.col_idx.zero_()
        self.val.zero_()
        wp.launch(fill_indices, dim=res, inputs=[self.rho0, self.row_idx, self.col_idx, self.val])
        self.L = sparse.bsr_from_triplets(res_flat, res_flat, self.row_idx, self.col_idx, self.val)
        
        # with wp.ScopedTimer("pressure solve"):
        self.p.zero_()
        linear.cg(self.L, self.u_div.flatten(), self.p.flatten(), tol=1e-6)
        wp.launch(substract_pressure, dim=res, inputs=[self.rho0, self.u0, self.p])

    def step(self):
        for _ in range(self.substeps):
            speed = 80.0
            angle = wp.sin(self.elapsed_time * 4.0) * wp.pi / 4 + wp.pi / 2
            vel = wp.vec2(wp.cos(angle) * speed, wp.sin(angle) * speed)
            
            wp.launch(update_emitter, dim=res_pad, inputs=[self.rho0, self.temp0, self.u0, 10.0, vel])
            wp.launch(apply_buoyancy, dim=res, inputs=[self.rho0, self.temp0, self.u0, self.sim_dt])
            self.pressure_project()
            wp.launch(advect, dim=res_pad, inputs=[self.u0, self.u1, self.rho0, self.rho1, self.temp0, self.temp1, self.sim_dt])
            
            (self.u0, self.u1) = (self.u1, self.u0)
            (self.rho0, self.rho1) = (self.rho1, self.rho0)
            (self.temp0, self.temp1) = (self.temp1, self.temp0)
            self.elapsed_time += self.sim_dt

    def step_and_render_frame(self, frame_num=None, img=None):
        with wp.ScopedTimer("step"):
            self.step()
        if img:
            img.set_array(self.rho0.numpy().T)
        return (img,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default="cuda", help="Override the default Warp device.")
    parser.add_argument("--num_frames", type=int, default=60*15, help="Total number of frames.")
    parser.add_argument("--output", type=str, default=None, help="Output video file.")

    args = parser.parse_known_args()[0]
    
    with wp.ScopedDevice(args.device):
        smoke = Smoke()
        import matplotlib
        import matplotlib.animation as anim
        import matplotlib.pyplot as plt

        fig = plt.figure()

        img = plt.imshow(
            smoke.rho0.numpy().T,
            origin="lower",
            animated=True,
            interpolation="antialiased",
        )
        img.set_norm(matplotlib.colors.Normalize(0.0, 2.508))
        seq = anim.FuncAnimation(
            fig,
            smoke.step_and_render_frame,
            fargs=(img,),
            frames=args.num_frames,
            blit=True,
            interval=8,
            repeat=False,
        )
        if args.output:
            writer = anim.FFMpegWriter(fps=60, metadata=dict(artist='Me'), bitrate=1800)
            seq.save(args.output, writer=writer)
            print(f"Video saved to {args.output}")
        else:
            plt.show()

# -------------------------------------------- not used --------------------------------------------
        
