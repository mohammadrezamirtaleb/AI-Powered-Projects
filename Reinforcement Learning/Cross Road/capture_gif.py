import os
import pygame
import imageio
import cv2
from src.main import Simulation

def capture_simulation_gif(output_path, duration=6.0, fps=12):
    sim = Simulation()
    # Speed up sim for better action
    sim.set_sim_speed(2.0)
    sim.set_ai_mode('MASTER')  # Show it working well
    
    frames = []
    clock = pygame.time.Clock()
    running = True
    
    # Let it warm up for a few seconds first
    print("Warming up...")
    for _ in range(60 * 3): # 3 seconds at 60 FPS
        dt = clock.tick(60) / 1000.0
        pygame.event.pump()
        sim.step_simulation(1.0 / 60.0)
    
    print("Capturing frames...")
    frames_to_capture = int(duration * fps)
    capture_interval = 60 // fps
    frame_count = 0
    captured = 0
    
    while running and captured < frames_to_capture:
        dt = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        sim.step_simulation(1.0 / 60.0)
        
        # Render everything
        sim.renderer.render_environment(sim.screen, sim.night_factor)
        sim.particle_mgr.draw_skids(sim.screen)
        sim.renderer.render_traffic_lights(sim.screen, sim.traffic_controller, sim.intersection.light_poles, sim.night_factor)
        is_night_bool = (sim.night_factor > 0.35)
        sim.pedestrian_mgr.draw(sim.screen, is_night=is_night_bool)
        for car in sim.vehicles:
            car.draw(sim.screen, is_night=is_night_bool, is_selected=False)
        sim.particle_mgr.draw_particles(sim.screen)
        
        light_dict = {
            'N': sim.traffic_controller.get_light_state('N'),
            'S': sim.traffic_controller.get_light_state('S'),
            'E': sim.traffic_controller.get_light_state('E'),
            'W': sim.traffic_controller.get_light_state('W')
        }
        sim.lighting.render_lighting(
            sim.screen, sim.vehicles, light_dict,
            sim.intersection.light_poles, sim.particle_mgr.particles,
            sim.night_factor
        )
        sim.weather.draw(sim.screen)
        
        # Draw HUD
        sim.hud.draw(sim.screen)
        pygame.display.flip()
        
        # Capture frame
        frame_count += 1
        if frame_count % capture_interval == 0:
            raw_str = pygame.image.tostring(sim.screen, 'RGB')
            image = pygame.image.fromstring(raw_str, sim.screen.get_size(), 'RGB')
            view = pygame.surfarray.array3d(image)
            view = view.transpose([1, 0, 2])
            
            # Resize the image to make GIF smaller (e.g., 800x450)
            resized = cv2.resize(view, (800, 450), interpolation=cv2.INTER_AREA)
            
            frames.append(resized)
            captured += 1
            print(f"Captured {captured}/{frames_to_capture}")
            
    pygame.quit()
    
    print("Saving GIF...")
    # save as gif
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"GIF saved to {output_path}")

if __name__ == "__main__":
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets', 'banner_animated.gif'))
    capture_simulation_gif(output_file)
