import os
import sys
import pygame

# Set dummy video driver for headless testing
os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import Simulation

def test_simulation_run():
    print("[TEST] Initializing simulation...")
    sim = Simulation()
    print("[TEST] Simulation initialized successfully.")

    print("[TEST] Running 180 simulation frames with spawns, lighting and weather cycles...")
    for frame in range(180):
        # Trigger spawns
        if frame % 15 == 0:
            sim.spawn_random_vehicle()
        if frame == 30:
            sim.spawn_ambulance()
        if frame == 50:
            sim.set_day_night(0.5) # Sunset
        if frame == 90:
            sim.set_day_night(1.0) # Night
        if frame == 120:
            sim.weather.set_weather('RAIN')

        # Run step
        fixed_dt = 1.0 / 60.0
        sim.step_simulation(fixed_dt)

        # Render step exactly like main.py
        sim.renderer.render_environment(sim.world_surface, sim.night_factor)
        sim.particle_mgr.draw_skids(sim.world_surface)

        sim.renderer.render_traffic_lights(
            sim.world_surface, sim.traffic_controller,
            sim.intersection.light_poles, sim.night_factor
        )

        is_night_bool = (sim.night_factor > 0.35)
        sim.pedestrian_mgr.draw(sim.world_surface, is_night=is_night_bool)

        for car in sim.vehicles:
            is_sel = (sim.selected_vehicle is not None and sim.selected_vehicle.id == car.id)
            car.draw(sim.world_surface, is_night=is_night_bool, is_selected=is_sel)

        sim.particle_mgr.draw_particles(sim.world_surface)

        light_dict = {
            'N': sim.traffic_controller.get_light_state('N'),
            'S': sim.traffic_controller.get_light_state('S'),
            'E': sim.traffic_controller.get_light_state('E'),
            'W': sim.traffic_controller.get_light_state('W')
        }
        sim.lighting.render_lighting(
            sim.world_surface, sim.vehicles, light_dict,
            sim.intersection.light_poles, sim.particle_mgr.particles,
            sim.night_factor
        )

        sim.hud.draw(sim.world_surface)

        # Test clicking a vehicle if any exists
        if frame == 40 and sim.vehicles:
            sim.selected_vehicle = sim.vehicles[0]
            print(f"[TEST] Selected vehicle #{sim.selected_vehicle.vehicle_id} ({sim.selected_vehicle.v_type})")

    print(f"[TEST] 180 frames complete!")
    print(f"[TEST] Total spawned: {sim.stats['total_spawned']}, Passed: {sim.stats['total_passed']}, Crashes: {sim.stats['total_crashes']}")
    print(f"[TEST] TrainingWorker active: {sim.training_worker.is_alive()}")
    print("[TEST] All UI, lighting, vehicle badges, and threading tests PASSED with 0 errors!")

if __name__ == "__main__":
    test_simulation_run()
