import numpy as np
from pycaw.pycaw import AudioUtilities


class VolumeController:
    def __init__(self):
        devices = AudioUtilities.GetSpeakers()
        self.volume = devices.EndpointVolume
        vol_range = self.volume.GetVolumeRange()
        self.min_vol = vol_range[0]
        self.max_vol = vol_range[1]

    def set_volume_percent(self, percent):
        vol = self.min_vol + (percent / 100.0) * (self.max_vol - self.min_vol)
        self.volume.SetMasterVolumeLevel(vol, None)

    def get_volume_percent(self):
        current = self.volume.GetMasterVolumeLevel()
        return int(np.interp(current, [self.min_vol, self.max_vol], [0, 100]))
