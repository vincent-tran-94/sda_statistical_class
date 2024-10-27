from dataclasses import dataclass
from enum import Enum


class AlternativeStudentHypothesis(Enum):
    TWO_SIDED = "two-sided"
    LESS = "less"
    GREATER = "greater"


@dataclass
class TtestInputTestParameters:
    equal_var: bool = True
    alternative: AlternativeStudentHypothesis = AlternativeStudentHypothesis.TWO_SIDED
