"""
A generic, score-based manager for tracking the health of any resource.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Set, Any

@dataclass
class ResourceHealth:
    """
    Tracks health metrics for a single resource using a score-based system.
    """
    resource_id: Any
    health_score: int = 100

class ResourceManager:
    """
    Tracks resource reliability using a heuristic health score.
    """
    STARTING_SCORE = 100
    SUCCESS_SCORE_INCREASE = 1
    FAILURE_SCORE_DECREASE = 10
    HEALTH_THRESHOLD = 50
    
    def __init__(self):
        self._health_records: Dict[Any, ResourceHealth] = {}

    def _get_or_create_record(self, resource_id: Any) -> ResourceHealth:
        """Gets an existing health record or creates a new one."""
        if resource_id not in self._health_records:
            self._health_records[resource_id] = ResourceHealth(
                resource_id=resource_id, 
                health_score=self.STARTING_SCORE
            )
        return self._health_records[resource_id]

    def record_failure(self, resource_id: Any):
        """Records a failure for a given resource, decreasing its health score."""
        record = self._get_or_create_record(resource_id)
        record.health_score -= self.FAILURE_SCORE_DECREASE

    def record_success(self, resource_id: Any):
        """Records a success for a given resource, increasing its health score."""
        record = self._get_or_create_record(resource_id)
        record.health_score = min(
            self.STARTING_SCORE, 
            record.health_score + self.SUCCESS_SCORE_INCREASE
        )

    def get_available_resources(self, all_resources: List[Any]) -> Set[Any]:
        """
        Returns a set of resources with a health score at or above the threshold.
        """
        available = set()
        for resource_id in all_resources:
            record = self._get_or_create_record(resource_id)
            if record.health_score >= self.HEALTH_THRESHOLD:
                available.add(resource_id)
        return available

    def select_resource(self, resource_list: List[Any]) -> Any:
        """
        Selects a single resource from a list using weighted random selection
        based on its health score.
        """
        if not resource_list:
            raise ValueError("Cannot select a resource from an empty list.")

        scores = [self._get_or_create_record(fp).health_score for fp in resource_list]
        weights = [max(0, score) for score in scores]

        if all(w == 0 for w in weights):
            return random.choice(resource_list)

        selected_resource = random.choices(resource_list, weights=weights, k=1)[0]
        return selected_resource
