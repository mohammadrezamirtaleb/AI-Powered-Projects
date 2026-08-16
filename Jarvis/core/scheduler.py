from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
import datetime
from .memory_vault import vault

class JarvisScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.notification_callbacks = []

    def start(self):
        self.scheduler.start()
        
    def stop(self):
        self.scheduler.shutdown()

    def add_notification_callback(self, callback):
        self.notification_callbacks.append(callback)
        
    async def _notify(self, message: str, title: str = "Notification"):
        for cb in self.notification_callbacks:
            await cb({"type": "notification", "title": title, "message": message})

    def schedule_reminder(self, title: str, message: str, run_date: datetime.datetime):
        async def reminder_job():
            await self._notify(message, title)
            vault.add_note(f"Reminder Triggered: {title}", message)
        
        self.scheduler.add_job(reminder_job, trigger=DateTrigger(run_date=run_date))
        return True

    def schedule_recurring_check(self, title: str, callback, minutes: int = 60):
        async def recurring_job():
            result = await callback()
            if result:
                await self._notify(result, title)
                
        self.scheduler.add_job(recurring_job, trigger=IntervalTrigger(minutes=minutes))

scheduler = JarvisScheduler()
