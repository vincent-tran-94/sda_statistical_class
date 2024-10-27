from dataclasses import dataclass, field
from typing import Tuple, Optional

from numpy._typing import ArrayLike


@dataclass
class ProblemContext:
    problem_statement: str
    data: Tuple[ArrayLike, ArrayLike]
    hints: Optional[list[str]] = field(default=None)
    solution: Optional[any] = field(default=None)

    def get_hints(self):
        return f"Hints available:\n- " + "\n- ".join(self.hints)

    def get_data(self):
        return self.data
