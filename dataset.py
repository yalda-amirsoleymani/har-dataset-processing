from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np


# In second
SAMPLE_RATE = 0.01


class ActivityType(Enum):
    WALKING = 1
    DESCENDING = 2
    ASCENDING = 3
    DRIVING = 4
    CLAPPING = 77
    UNKNOWN = 99


class Sensor(Enum):
    LEFT_WRIST = 0
    LEFT_HIP = 1
    LEFT_ANKLE = 2
    RIGHT_ANKLE = 3


SENSOR_AXES = {
    Sensor.LEFT_WRIST: ["lw_x", "lw_y", "lw_z"],
    Sensor.LEFT_HIP: ["lh_x", "lh_y", "lh_z"],
    Sensor.LEFT_ANKLE: ["la_x", "la_y", "la_z"],
    Sensor.RIGHT_ANKLE: ["ra_x", "ra_y", "ra_z"],
}


class Dataset:
    def __init__(self, path_str: str):
        """
        path: Path to the directory containing all CSV files
        """
        paths = Path(path_str).glob("*.csv")
        self._df = None
        for i, p in enumerate(paths):
            self._df = pd.concat([self._df, pd.read_csv(p)])
            # TODO: remove
            #if i > 5:
            #    break

        # Add a new column timestamp
        self._df["timestamp"] = 0

        self._acts = []
        for typ in ActivityType:
            # Set increasing timestamp for every single activity
            msk = self._df["activity"] == typ.value
            self._df.loc[msk, "timestamp"] = np.arange(self._df[msk].shape[0])

    def get(self, typ: ActivityType, sensor: Sensor, start=0, duration=-1):
        """
        Retrive sample data bases on activity type and sensor position

        Arguments:
            typ: Type of activity
            sensor: sensor position
            start: start time in second
            duration: duration in second

        Return:
            Numpy array of XYZ axes shape: (n, 3)
            n is number of samples
        """
        axes = SENSOR_AXES[sensor]
        df = self._df[self._df["activity"] == typ.value][axes]
        start = int(start / SAMPLE_RATE)
        if duration == -1:
            end = -1
        else:
            end = start + int(duration / SAMPLE_RATE)
        return df.iloc[start:end].to_numpy()
