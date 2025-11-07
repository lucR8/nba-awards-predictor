from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class MVPConfig:
    numeric_features: List[str] = (
        "gp mpg pts ast reb ts_pct usg_pct ws bpm vorp team_win_pct off_rating def_rating".split()
    )
    categorical_features: List[str] = ("position team starter_flag".split())
