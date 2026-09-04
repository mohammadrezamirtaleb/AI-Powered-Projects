import os
import sys
os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import Simulation

def test_resize():
    print("[TEST] Initializing simulation for resize test...")
    sim = Simulation()
    print(f"[TEST] Initial resolution: {sim.width}x{sim.height}")
    
    # Test resizing to 1920x1080
    sim.resize_world(1920, 1080)
    assert sim.width == 1920
    assert sim.height == 1080
    assert sim.world_surface.get_size() == (1920, 1080)
    print(f"[TEST] Resized to 1920x1080 successfully. Canvas center: ({sim.intersection.cx}, {sim.intersection.cy})")
    
    # Test resizing to 1536x864 (user resolution)
    sim.resize_world(1536, 864)
    assert sim.width == 1536
    assert sim.height == 864
    assert sim.world_surface.get_size() == (1536, 864)
    print(f"[TEST] Resized to 1536x864 successfully.")

    # Render a frame at 1536x864
    sim.renderer.render_environment(sim.world_surface, sim.night_factor)
    sim.hud.draw(sim.world_surface)
    print("[TEST] Rendered at 1536x864 with 0 errors!")

if __name__ == "__main__":
    test_resize()
