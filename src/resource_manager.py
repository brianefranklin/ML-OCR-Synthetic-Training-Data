"""A generic, score-based manager for tracking the health of any resource.

This module provides a reusable ResourceManager class that can be extended to
manage the health and selection of any type of resource (e.g., fonts, backgrounds).
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Set, Any, TypeVar

T = TypeVar('T')

@dataclass
class ResourceHealth:
    """Tracks health metrics for a single resource.

    Attributes:
        resource_id: The unique identifier for the resource.
        health_score: The current health score of the resource.
    """
    resource_id: Any
    health_score: int = 100

class ResourceManager:
    """Tracks resource reliability using a heuristic health score.

    This class provides a system for dynamically scoring resources based on their
    success or failure in generation tasks. It allows for filtering out unhealthy
    resources and selecting from healthy ones with a weighted probability.
    """
    STARTING_SCORE: int = 100
    SUCCESS_SCORE_INCREASE: int = 1
    FAILURE_SCORE_DECREASE: int = 10
    HEALTH_THRESHOLD: int = 50
    
    def __init__(self):
        """Initializes the ResourceManager."""
        self._health_records: Dict[Any, ResourceHealth] = {}

    def _get_or_create_record(self, resource_id: Any) -> ResourceHealth:
        """Gets an existing health record or creates a new one if it doesn't exist.

        Args:
            resource_id: The unique identifier for the resource.

        Returns:
            The ResourceHealth object for the given resource.
        """
        if resource_id not in self._health_records:
            self._health_records[resource_id] = ResourceHealth(
                resource_id=resource_id, 
                health_score=self.STARTING_SCORE
            )
        return self._health_records[resource_id]

    def record_failure(self, resource_id: Any) -> None:
        """Records a failure for a given resource, decreasing its health score."""
        record = self._get_or_create_record(resource_id)
        record.health_score -= self.FAILURE_SCORE_DECREASE

    def record_success(self, resource_id: Any) -> None:
        """Records a success for a given resource, increasing its health score.
        
        The score is capped at the STARTING_SCORE.
        """
        record = self._get_or_create_record(resource_id)
        record.health_score = min(
            self.STARTING_SCORE, 
            record.health_score + self.SUCCESS_SCORE_INCREASE
        )

    def get_available_resources(self, all_resources: List[T]) -> Set[T]:
        """Returns a set of resources with a health score at or above the threshold.

        Args:
            all_resources: A list of all resources to check.

        Returns:
            A set of the resources that are considered healthy.
        """
        available: Set[T] = set()
        for resource_id in all_resources:
            record = self._get_or_create_record(resource_id)
            if record.health_score >= self.HEALTH_THRESHOLD:
                available.add(resource_id)
        return available

    def select_resource(self, resource_list: List[T]) -> T:
        """Selects a single resource from a list using weighted random selection.

        The selection is weighted based on the health score of each resource.

        Args:
            resource_list: The list of resources to select from.

        Returns:
            A single resource selected from the list.
            
        Raises:
            ValueError: If the resource_list is empty.
        """
        if not resource_list:
            raise ValueError("Cannot select a resource from an empty list.")

        # Get the health scores for each resource in the list.
        scores = [self._get_or_create_record(fp).health_score for fp in resource_list]
        # Use max(0, score) to ensure weights are not negative.
        weights = [max(0, score) for score in scores]

        # If all resources have a score of 0, choose one uniformly.
        if all(w == 0 for w in weights):
            return random.choice(resource_list)

        # Use random.choices for weighted selection.
        selected_resource = random.choices(resource_list, weights=weights, k=1)[0]
        return selected_resource