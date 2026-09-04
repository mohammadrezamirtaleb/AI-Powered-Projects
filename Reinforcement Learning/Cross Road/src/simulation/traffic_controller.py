"""
Traffic Light Phase Controller for 4-way intersection.
Manages automatic green-yellow-red cycling, all-red clearance intervals,
emergency vehicle preemption, and actuated / adaptive traffic queue optimization.
"""
from src.config import TRAFFIC_LIGHT_PHASES, PHASE_DURATIONS

class TrafficController:
    def __init__(self):
        self.phase_keys = ['NS_GREEN', 'NS_YELLOW', 'ALL_RED_1', 'EW_GREEN', 'EW_YELLOW', 'ALL_RED_2']
        self.current_phase_index = 0
        self.timer = 0.0
        self.is_manual = False
        self.adaptive_mode = False
        self.emergency_override = False
        self.preemption_cooldown = 0.0

    @property
    def current_phase(self):
        return self.phase_keys[self.current_phase_index]

    def update(self, dt, vehicles=None):
        """Update traffic light timer by dt with robust adaptive and emergency control."""
        if self.is_manual:
            return

        if self.preemption_cooldown > 0:
            self.preemption_cooldown = max(0.0, self.preemption_cooldown - dt)

        # 1. Emergency Vehicle Preemption Check (only if not already in yellow/all-red transition)
        cur_phase = self.current_phase
        if vehicles and self.preemption_cooldown <= 0.0 and cur_phase in ('NS_GREEN', 'EW_GREEN'):
            ns_emergency = False
            ew_emergency = False
            for v in vehicles:
                if v.is_alive and getattr(v, 'is_emergency', False):
                    dist = v.get_distance_to_stop_line()
                    if 0.0 < dist < 200.0:
                        if v.route.start_dir in ('N', 'S'):
                            ns_emergency = True
                        elif v.route.start_dir in ('E', 'W'):
                            ew_emergency = True

            # If emergency is on opposing green, initiate standard yellow clearance
            if ns_emergency and cur_phase == 'EW_GREEN' and self.timer > 3.0:
                self.current_phase_index = 4 # EW_YELLOW
                self.timer = 0.0
                self.preemption_cooldown = 8.0
            elif ew_emergency and cur_phase == 'NS_GREEN' and self.timer > 3.0:
                self.current_phase_index = 1 # NS_YELLOW
                self.timer = 0.0
                self.preemption_cooldown = 8.0

        self.timer += dt
        cur_phase = self.current_phase
        max_duration = PHASE_DURATIONS.get(cur_phase, 8.0)

        # 2. Adaptive Queue Optimization
        if self.adaptive_mode and vehicles and cur_phase in ('NS_GREEN', 'EW_GREEN'):
            ns_queue = sum(1 for v in vehicles if v.is_alive and v.route.start_dir in ('N', 'S') and 0 < v.get_distance_to_stop_line() < 180)
            ew_queue = sum(1 for v in vehicles if v.is_alive and v.route.start_dir in ('E', 'W') and 0 < v.get_distance_to_stop_line() < 180)

            if cur_phase == 'NS_GREEN':
                # Extend if heavy NS traffic, cut early if empty and EW waiting
                if ns_queue > 3 and ew_queue < 2:
                    max_duration = 16.0
                elif ns_queue == 0 and ew_queue >= 2 and self.timer > 4.0:
                    max_duration = 4.0
            elif cur_phase == 'EW_GREEN':
                if ew_queue > 3 and ns_queue < 2:
                    max_duration = 16.0
                elif ew_queue == 0 and ns_queue >= 2 and self.timer > 4.0:
                    max_duration = 4.0

        if self.timer >= max_duration:
            self.timer = 0.0
            self.current_phase_index = (self.current_phase_index + 1) % len(self.phase_keys)

    def get_light_state(self, direction):
        """
        Get light state ('RED', 'YELLOW', 'GREEN') for a direction ('N', 'S', 'E', 'W').
        """
        phase = self.current_phase

        if direction in ('N', 'S'):
            if phase == 'NS_GREEN':
                return 'GREEN'
            elif phase == 'NS_YELLOW':
                return 'YELLOW'
            else:
                return 'RED'
        elif direction in ('E', 'W'):
            if phase == 'EW_GREEN':
                return 'GREEN'
            elif phase == 'EW_YELLOW':
                return 'YELLOW'
            else:
                return 'RED'

        return 'RED'

    def toggle_manual(self):
        """Toggle manual override mode."""
        self.is_manual = not self.is_manual
        if self.is_manual:
            self.current_phase_index = 0
            self.timer = 0.0

    def toggle_adaptive(self):
        """Toggle actuated / adaptive traffic controller mode."""
        self.adaptive_mode = not self.adaptive_mode

    def switch_manual_phase(self):
        """Manually trigger next major green phase."""
        if not self.is_manual:
            self.is_manual = True

        if self.current_phase in ('NS_GREEN', 'NS_YELLOW', 'ALL_RED_1'):
            self.current_phase_index = 3 # EW_GREEN
        else:
            self.current_phase_index = 0 # NS_GREEN
        self.timer = 0.0

    def get_time_remaining(self):
        """Return remaining seconds in current phase."""
        if self.is_manual:
            return 99.0
        cur_phase = self.current_phase
        max_duration = PHASE_DURATIONS.get(cur_phase, 8.0)
        return max(0.0, max_duration - self.timer)
