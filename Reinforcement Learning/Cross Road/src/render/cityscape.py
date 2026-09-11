"""
2.5D city dressing baked into the static world: extruded curbs, building
masses, tree canopies, and a landscaped roundabout plaza.

Everything here is drawn once per day/night cache so the live loop stays at 60 FPS.
"""
import math
import random
import pygame


def _clamp_rgb(c):
    return (max(0, min(255, int(c[0]))), max(0, min(255, int(c[1]))), max(0, min(255, int(c[2]))))


def _shade(c, d):
    return _clamp_rgb((c[0] + d, c[1] + d, c[2] + d))


def is_road_cell(x, y, cx, cy, rbx, rby, hrw, r_outer, r_island):
    on_ns = abs(x - cx) <= hrw + 18
    on_ew = abs(y - cy) <= hrw + 18
    on_blvd = abs(y - rby) <= hrw + 18
    on_ring = math.hypot(x - rbx, y - rby) <= r_outer + 22
    return on_ns or on_ew or on_blvd or on_ring


def draw_extruded_curb(surface, rect, night):
    x, y, w, h = rect
    lip = (58, 62, 68) if night else (168, 172, 178)
    shade = (28, 30, 34) if night else (92, 96, 102)
    pygame.draw.line(surface, shade, (x + 1, y + h), (x + w, y + h), 3)
    pygame.draw.line(surface, shade, (x + w, y + 1), (x + w, y + h), 3)
    pygame.draw.line(surface, lip, (x, y), (x + w, y), 2)
    pygame.draw.line(surface, lip, (x, y), (x, y + h), 2)


def draw_building(surface, x, y, w, h, depth, wall, roof, night):
    """Simple isometric-ish mass: floor shadow, south/east walls, roof, windows."""
    shadow = [(x + 6, y + h + 4), (x + w + 6, y + h + 4),
              (x + w + depth + 6, y + h - depth + 4), (x + depth + 6, y + h - depth + 4)]
    pygame.draw.polygon(surface, (12, 16, 20) if night else (48, 72, 42), shadow)

    south = _shade(wall, -28)
    east = _shade(wall, -18)
    pygame.draw.polygon(surface, south, [
        (x, y + h), (x + w, y + h), (x + w + depth, y + h - depth), (x + depth, y + h - depth)
    ])
    pygame.draw.polygon(surface, east, [
        (x + w, y), (x + w + depth, y - depth), (x + w + depth, y + h - depth), (x + w, y + h)
    ])
    pygame.draw.rect(surface, roof, (x, y - depth, w, h))
    pygame.draw.rect(surface, _shade(roof, 18), (x, y - depth, w, 4))
    pygame.draw.rect(surface, _shade(wall, -40), (x, y - depth, w, h), 1)

    win = (255, 214, 140) if night else (70, 92, 118)
    dark = (28, 24, 18) if night else (48, 58, 70)
    cols = max(2, w // 14)
    rows = max(2, h // 16)
    inset_x = 6
    inset_y = 8
    cell_w = max(4, (w - inset_x * 2) // cols - 2)
    cell_h = max(3, (h - inset_y * 2) // rows - 3)
    for r in range(rows):
        for c in range(cols):
            wx = x + inset_x + c * (cell_w + 4)
            wy = y - depth + inset_y + r * (cell_h + 5)
            lit = (not night) or ((r + c + int(x)) % 3 != 0)
            pygame.draw.rect(surface, win if lit else dark, (wx, wy, cell_w, cell_h), border_radius=1)


def draw_tree(surface, x, y, r, night):
    pygame.draw.ellipse(surface, (18, 28, 16) if night else (46, 78, 38),
                        (x - r, y + r * 0.35, r * 2, r * 0.7))
    trunk = (48, 32, 22) if night else (92, 62, 40)
    pygame.draw.rect(surface, trunk, (x - 2, y, 4, int(r * 0.7)))
    canopy = (28, 72, 40) if night else (46, 122, 62)
    hi = _shade(canopy, 28)
    pygame.draw.circle(surface, canopy, (x, y - int(r * 0.15)), r)
    pygame.draw.circle(surface, hi, (x - r // 3, y - r // 2), max(4, r // 3))


def draw_roundabout_plaza(surface, rbx, rby, r_island, r_outer, night, r_inner=48.0, r_circ=74.0):
    cobble = (42, 44, 48) if night else (72, 74, 80)
    pygame.draw.circle(surface, cobble, (int(rbx), int(rby)), int(r_outer))
    ring = (210, 214, 220) if night else (238, 240, 244)
    pygame.draw.circle(surface, ring, (int(rbx), int(rby)), int(r_outer), 2)
    split_r = int((r_inner + r_circ) * 0.5)
    dash_col = (200, 204, 210) if night else (228, 230, 236)
    for i in range(0, 360, 14):
        a0 = math.radians(i)
        a1 = math.radians(i + 7)
        p0 = (rbx + split_r * math.cos(a0), rby + split_r * math.sin(a0))
        p1 = (rbx + split_r * math.cos(a1), rby + split_r * math.sin(a1))
        pygame.draw.line(surface, dash_col, p0, p1, 2)
    for rr in (r_inner, r_circ):
        for i in range(10):
            a = i * (2 * math.pi / 10) - math.pi / 2
            cx = rbx + math.cos(a) * rr
            cy = rby + math.sin(a) * rr
            tang = a + math.pi / 2
            p0 = (cx + math.cos(tang) * 6, cy + math.sin(tang) * 6)
            p1 = (cx - math.cos(tang) * 4 + math.cos(a) * 3, cy - math.sin(tang) * 4 + math.sin(a) * 3)
            p2 = (cx - math.cos(tang) * 4 - math.cos(a) * 3, cy - math.sin(tang) * 4 - math.sin(a) * 3)
            pygame.draw.polygon(surface, (232, 186, 64), [p0, p1, p2])

    island = (32, 58, 38) if night else (74, 132, 78)
    pygame.draw.circle(surface, (22, 26, 30) if night else (70, 74, 78), (int(rbx) + 3, int(rby) + 4), int(r_island))
    pygame.draw.circle(surface, island, (int(rbx), int(rby)), int(r_island))
    pygame.draw.circle(surface, (186, 190, 196) if night else (220, 224, 230), (int(rbx), int(rby)), int(r_island), 3)
    pygame.draw.circle(surface, (48, 92, 118) if night else (92, 148, 176), (int(rbx), int(rby)), 11)
    pygame.draw.circle(surface, (160, 210, 230) if night else (200, 230, 240), (int(rbx), int(rby)), 5)
    pygame.draw.circle(surface, (240, 248, 255), (int(rbx) - 2, int(rby) - 2), 2)


def paint_parks_and_city(surface, cx, cy, rbx, rby, hrw, r_outer, r_island,
                         canvas_right, canvas_top, canvas_bottom, night):
    random.seed(19)
    # Soft park blobs in grass
    park = (28, 58, 36) if night else (64, 118, 70)
    for ox, oy, rw, rh in (
        (90, canvas_top + 70, 110, 70),
        (canvas_right - 160, canvas_top + 80, 120, 64),
        (70, (cy + rby) * 0.5, 90, 80),
        (canvas_right - 140, (cy + rby) * 0.52, 100, 76),
        (80, canvas_bottom - 90, 100, 50),
        (canvas_right - 150, canvas_bottom - 95, 110, 55),
    ):
        pygame.draw.ellipse(surface, park, (ox, oy, rw, rh))

    buildings = [
        (cx - 250, cy - 210, 78, 52, 14, (92, 78, 68), (138, 92, 72)),
        (cx - 330, cy - 150, 64, 70, 12, (70, 82, 96), (118, 128, 140)),
        (cx + 175, cy - 200, 86, 48, 16, (86, 74, 90), (150, 118, 128)),
        (cx + 270, cy - 155, 58, 62, 12, (74, 88, 78), (120, 140, 112)),
        (cx - 280, (cy + rby) * 0.5 - 20, 72, 44, 12, (96, 80, 64), (160, 120, 80)),
        (cx + 200, (cy + rby) * 0.5 - 10, 80, 50, 14, (68, 76, 92), (110, 122, 148)),
        (cx - 300, rby + 120, 70, 46, 12, (88, 70, 70), (148, 108, 100)),
        (cx + 210, rby + 115, 76, 50, 13, (72, 84, 70), (124, 140, 108)),
        (40, cy - 200, 54, 40, 10, (80, 86, 96), (130, 136, 148)),
        (canvas_right - 110, cy - 190, 58, 42, 10, (90, 78, 70), (150, 118, 96)),
    ]
    for x, y, w, h, d, wall, roof in buildings:
        bx, by = int(x), int(y)
        if is_road_cell(bx + w / 2, by + h / 2, cx, cy, rbx, rby, hrw, r_outer, r_island):
            continue
        if night:
            wall = _shade(wall, -40)
            roof = _shade(roof, -50)
        draw_building(surface, bx, by, w, h, d, wall, roof, night)


def paint_trees(surface, cx, cy, rbx, rby, hrw, r_outer, r_island,
                canvas_right, canvas_top, canvas_bottom, night):
    trees = [
        (cx - 120, cy - 175, 11), (cx + 118, cy - 168, 10),
        (cx - 130, cy + 175, 12), (cx + 128, cy + 170, 11),
        (cx - 200, rby - 130, 13), (cx + 205, rby - 125, 12),
        (cx - 190, rby + 128, 11), (cx + 198, rby + 132, 12),
        (60, canvas_top + 90, 10), (canvas_right - 70, canvas_top + 95, 11),
        (55, canvas_bottom - 70, 10), (canvas_right - 64, canvas_bottom - 74, 11),
        (rbx - 22, rby - 10, 7), (rbx + 20, rby + 8, 6),
    ]
    for tx, ty, tr in trees:
        on_ring = math.hypot(tx - rbx, ty - rby) <= r_outer + 8
        in_island = math.hypot(tx - rbx, ty - rby) < r_island - 2
        if is_road_cell(tx, ty, cx, cy, rbx, rby, hrw, r_outer, r_island) and not in_island:
            continue
        if on_ring and not in_island:
            continue
        draw_tree(surface, int(tx), int(ty), tr, night)
