from enum import Enum

class SunPosition(Enum):
    FIXED = 1
    SAMPLE_ONCE = 2
    SAMPLE_EVERY_VIEW = 3

class SpotlightsPosition(Enum):
    THREESPOTLIGHTS = 1
    THREESPOTLIGHTS_SAMPLED = 2
    AMBIENT_LIGHT = 3