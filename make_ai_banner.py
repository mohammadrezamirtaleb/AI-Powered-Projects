import pygame
import imageio
import numpy as np
import math
import cv2
import os

def render_glowing_line(surface, color, start, end, thickness, bloom=True):
    pygame.draw.line(surface, color, start, end, thickness)
    if bloom:
        pygame.draw.line(surface, (color[0]//2, color[1]//2, color[2]//2), start, end, thickness * 2)
        pygame.draw.line(surface, (color[0]//4, color[1]//4, color[2]//4), start, end, thickness * 4)

def main():
    width, height = 1200, 400
    pygame.init()
    surface = pygame.Surface((width, height))
    
    # Define network layout
    layers = [8, 12, 12, 6]
    layer_xs = [200, 466, 733, 1000]
    
    nodes = []
    for i, count in enumerate(layers):
        layer_nodes = []
        spacing = (height - 100) / (count - 1) if count > 1 else 0
        start_y = 50
        for j in range(count):
            layer_nodes.append((layer_xs[i], start_y + j * spacing))
        nodes.append(layer_nodes)
        
    frames = []
    fps = 30
    duration = 4.0
    total_frames = int(fps * duration)
    
    # Pre-calculate connections
    connections = []
    for i in range(len(nodes) - 1):
        for n1 in nodes[i]:
            for n2 in nodes[i+1]:
                connections.append((n1, n2))
                
    for f in range(total_frames):
        surface.fill((10, 12, 20)) # Dark cyber background
        
        time_t = f / total_frames
        
        # Draw connections
        for idx, (p1, p2) in enumerate(connections):
            # Base line
            pygame.draw.line(surface, (30, 40, 60), p1, p2, 1)
            
            # Pulse
            pulse_offset = (time_t * 3.0 + (idx * 0.05)) % 1.0
            if pulse_offset < 0.2:
                intensity = math.sin(pulse_offset * math.pi * 5)
                if intensity > 0:
                    px = p1[0] + (p2[0] - p1[0]) * pulse_offset * 5
                    py = p1[1] + (p2[1] - p1[1]) * pulse_offset * 5
                    
                    # Draw a small glowing trail
                    trail_start = (px - (p2[0] - p1[0])*0.05, py - (p2[1] - p1[1])*0.05)
                    pygame.draw.line(surface, (0, 200, 255), trail_start, (px, py), 2)
        
        # Draw nodes
        for i, layer_nodes in enumerate(nodes):
            for j, pos in enumerate(layer_nodes):
                glow_val = (math.sin(time_t * math.pi * 4 + i + j) + 1) / 2
                color = (0, int(150 + 105*glow_val), int(200 + 55*glow_val)) if i != len(nodes)-1 else (255, int(100+100*glow_val), 100)
                
                pygame.draw.circle(surface, color, (int(pos[0]), int(pos[1])), 6)
                pygame.draw.circle(surface, (color[0]//2, color[1]//2, color[2]//2), (int(pos[0]), int(pos[1])), 12, 1)
                pygame.draw.circle(surface, (color[0]//4, color[1]//4, color[2]//4), (int(pos[0]), int(pos[1])), 20, 1)
                
        # Draw some tech text
        font = pygame.font.SysFont("Consolas", 36, bold=True)
        txt = font.render("AI-POWERED PROJECTS", True, (0, 255, 255))
        txt.set_alpha(int(150 + 100 * math.sin(time_t * math.pi * 2)))
        surface.blit(txt, (420, 320))
        
        # Capture
        raw_str = pygame.image.tostring(surface, 'RGB')
        image = pygame.image.fromstring(raw_str, (width, height), 'RGB')
        view = pygame.surfarray.array3d(image).transpose([1, 0, 2])
        
        frames.append(view)
        print(f"Rendered frame {f+1}/{total_frames}")
        
    print("Saving GIF...")
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets', 'banner_ai_animated.gif'))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, duration=1000/fps, loop=0)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
