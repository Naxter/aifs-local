"""Input plugin reading states written by anemoi-inference's raw output.

The built-in ``raw`` output writes one compressed .npz per state, each
field stored under ``field_{name}`` next to ``date``, ``latitudes`` and
``longitudes`` arrays. This input reads those files back, so a forecast
can be replayed or continued from stored states without re-fetching the
original source data.
"""

import logging
from pathlib import Path

import earthkit.data as ekd
import numpy as np

from anemoi.inference.inputs.ekd import EkdInput
from anemoi.inference.types import Date
from anemoi.inference.types import State

LOG = logging.getLogger(__name__)

# grib_keys entries that would clash with the per-date metadata set below.
SKIP_KEYS = ["date", "time", "step", "valid_datetime"]


class RawInputPlugin(EkdInput):
    """Read input states from raw-output .npz files."""

    trace_name = "raw"

    def __init__(
        self,
        context,
        path,
        template: str = "{date}.npz",
        strftime: str = "%Y%m%d%H%M%S",
        **kwargs,
    ) -> None:
        super().__init__(context, **kwargs)
        self.path = Path(path)
        self.template = template
        self.strftime = strftime

    def __repr__(self) -> str:
        return f"RawInputPlugin({self.path})"

    def create_input_state(self, *, date: Date | None, **kwargs) -> State:
        assert date is not None, "date must be provided for the raw input"
        dates = [date + lag for lag in self.checkpoint.lagged]
        return self._create_input_state(
            self._fields(dates, self.variables), variables=None, date=date, **kwargs
        )

    def load_forcings_state(self, *, dates: list[Date], current_state: State) -> State:
        return self._load_forcings_state(
            self._fields(dates, self.variables), dates=dates, current_state=current_state
        )

    def _fields(self, dates: list[Date], variables: list[str]) -> ekd.FieldList:
        typed_variables = self.checkpoint.typed_variables

        result = []
        for date in dates:
            file = self.path / self.template.format(date=date.strftime(self.strftime))
            if not file.exists():
                raise FileNotFoundError(
                    f"{file} not found: the raw input needs a state file for every "
                    f"input date ({', '.join(str(d) for d in dates)})"
                )
            with np.load(file) as npz:
                latitudes = np.array(npz["latitudes"])
                longitudes = np.array(npz["longitudes"])
                for variable in variables:
                    key = f"field_{variable}"
                    if key not in npz.files:
                        raise KeyError(f"{key} not found in {file}")
                    grib_keys = {
                        k: v
                        for k, v in typed_variables[variable].grib_keys.items()
                        if k not in SKIP_KEYS
                    }
                    result.append(
                        dict(
                            values=np.array(npz[key]),
                            latitudes=latitudes,
                            longitudes=longitudes,
                            date=date.strftime("%Y%m%d"),
                            time=date.strftime("%H%M"),
                            name=variable,
                            **grib_keys,
                        )
                    )

        return ekd.from_source("list-of-dicts", result)

    def template_lookup(self, name: str) -> dict:
        # Required by the template manager; this input provides no templates.
        return {}
