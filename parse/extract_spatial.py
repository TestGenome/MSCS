from pysc2.lib.features import MINIMAP_FEATURES
import numpy as np


class SpatialParser:
    def __init__(self):
        self.features = ['height_map', 'pathable', 'visibility_map', 'creep', 'player_relative', 'unit_type']

    def extract(self, obs):
        return np.stack([f.unpack(obs) for f in MINIMAP_FEATURES if f.name in self.features])#.astype(np.float32, copy=False)

    def get_scale(self):
        return [(f.name,f.type.name,f.scale) for f in MINIMAP_FEATURES if f.name in self.features]