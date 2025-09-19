from PCISPH import PCISPH
from BaseSPH import *
import warp.render

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default="cuda", help="Override the default Warp device.")
    parser.add_argument("--num_frames", type=int, default=60*7, help="Total number of frames.")
    parser.add_argument("--output", type=str, default=None, help="Output video file.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]
    
    with wp.ScopedDevice(args.device):
        simulator = PCISPH()
        
        simulator.add_cube_emitter(wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 2.0, 0.3))
        # simulator.add_cube_emitter(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.2, 0.2, 0.2))
        # simulator.add_cube_emitter(wp.vec3(0.7, 0.7, 0.0), wp.vec3(1.0, 1.0, 1.0))
        simulator.add_collider(wp.vec3(0.5, 0.3, 1.0), 0.3)
        simulator.finalize()
        
        renderer = None
        if args.output:
            renderer = warp.render.UsdRenderer(args.output)
        else:
            renderer = warp.render.OpenGLRenderer(
                screen_width=1024,
                screen_height=1024,
                camera_pos=(0, 4, 4),
                camera_front=(0.4, -1, -1)
            )
        
        for num_frame in range(args.num_frames):
            num_steps = int(simulator.dt // simulator.sim_dt)
            for _ in range(num_steps):
                simulator.step(args.verbose)
            simulator.sim_dt = simulator.dt - num_steps * simulator.sim_dt
            simulator.step(args.verbose)

            with wp.ScopedTimer("render"):
                renderer.begin_frame(num_frame * simulator.dt)
                renderer.render_points(
                    name="points",
                    points=simulator.x.numpy(), 
                    radius=p_radius / 2
                )
                renderer.end_frame()
                
            simulator.compute_sim_dt()
            
        if args.output:
            renderer.save()