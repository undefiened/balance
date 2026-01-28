"""
File description:
-----------------
This module defines data structures for storing solution results from greedy and IP optimization methods
without modifying the original Intent objects.
"""
from typing import List, Dict, Union, Optional
import dataclasses

import utils


@dataclasses.dataclass
class IntentSolution:
    """Stores the solution for a single intent."""
    intent_key: tuple  # (source, destination, start)
    path: List[utils.Link]
    actual_time: int
    ideal_time: int
    solution_found: bool
    adjusted_start: Optional[int] = None  # For greedy, when start time is adjusted

    @property
    def time_difference(self) -> Optional[int]:
        """Returns the difference between actual and ideal time."""
        if self.solution_found and self.ideal_time > 0 and self.actual_time > 0:
            return self.actual_time - self.ideal_time
        return None


@dataclasses.dataclass
class GreedySolution:
    """Stores the complete greedy solution for all intents."""
    solutions: Dict[tuple, IntentSolution]  # Maps intent key to its solution
    total_objective: Optional[int]
    
    def get_solution(self, intent_key: tuple) -> Optional[IntentSolution]:
        """Get solution for a specific intent."""
        return self.solutions.get(intent_key)


@dataclasses.dataclass
class IPSolution:
    """Stores the complete IP solution for all intents."""
    solutions: Dict[tuple, IntentSolution]  # Maps intent key to its solution
    total_objective: Optional[float]
    model_gap: float
    
    def get_solution(self, intent_key: tuple) -> Optional[IntentSolution]:
        """Get solution for a specific intent."""
        return self.solutions.get(intent_key)


# =============================================== END OF FILE =============================================== 