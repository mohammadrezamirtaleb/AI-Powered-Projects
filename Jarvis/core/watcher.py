import psutil
from .scheduler import scheduler
from .memory_vault import vault

class SystemWatcher:
    def __init__(self):
        self.cpu_threshold = 90.0
        self.memory_threshold = 90.0

    async def check_vitals(self):
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        
        warnings = []
        if cpu > self.cpu_threshold:
            warnings.append(f"High CPU Usage: {cpu}%")
        if mem > self.memory_threshold:
            warnings.append(f"High Memory Usage: {mem}%")
            
        if warnings:
            message = " | ".join(warnings)
            vault.add_note("System Warning", message)
            return message
        return None

watcher = SystemWatcher()

def init_watcher():
    scheduler.schedule_recurring_check("System Vitals Warning", watcher.check_vitals, minutes=5)
