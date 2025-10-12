"""
Defines the data structures for batch configurations and provides a loader
for YAML-based configuration files.
"""
import yaml
from dataclasses import dataclass, field
from typing import List, Type, TypeVar, Dict

# A generic type for the from_yaml method, allowing it to be used in subclasses.
T = TypeVar('T')

@dataclass
class BatchSpecification:
    """
    Represents the configuration for a single batch within a larger generation task.

    Attributes:
        name (str): A unique identifier for the batch.
        proportion (float): The proportion of the total images that this batch
                            should represent.
        text_direction (str): The direction of the text rendering.
        corpus_file (str): The path to the corpus file to be used for this batch.
    """
    name: str
    proportion: float
    text_direction: str
    corpus_file: str

@dataclass
class BatchConfig:
    """
    Represents the entire batch configuration, including the total number of images
    and a list of individual batch specifications.

    Attributes:
        total_images (int): The total number of images to generate for this entire
                            batch run.
        specifications (List[BatchSpecification]): A list of individual batch
                                                   specifications.
    """
    total_images: int
    specifications: List[BatchSpecification] = field(default_factory=list)

    @classmethod
    def from_yaml(cls: Type[T], yaml_path: str) -> T:
        """
        Loads and parses a YAML file into a BatchConfig object.

        Args:
            yaml_path (str): The path to the YAML configuration file.

        Returns:
            An instance of the class populated with data from the YAML file.
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        spec_data = config_data.get('specifications', [])
        specifications = [BatchSpecification(**spec) for spec in spec_data]

        return cls(
            total_images=config_data.get('total_images', 0),
            specifications=specifications
        )

class BatchManager:
    """
    Manages the allocation of images to different batch specifications.
    """
    def __init__(self, config: BatchConfig):
        """
        Initializes the BatchManager with a BatchConfig.

        Args:
            config (BatchConfig): The batch configuration object.
        """
        self.config = config
        self._allocation: Dict[str, int] = {}
        self._allocate_images()

    def _allocate_images(self):
        """
        Allocates the total number of images to the different batches based
        on their proportions. Handles rounding and distributes remainders.
        """
        total_images = self.config.total_images
        specs = self.config.specifications

        # Initial allocation based on proportions, ignoring remainders
        allocated_counts = {spec.name: int(spec.proportion * total_images) for spec in specs}

        # Distribute the remainder
        remainder = total_images - sum(allocated_counts.values())

        # Distribute the remainder one by one to the specs.
        # To make the distribution deterministic, we sort the specs by name.
        for spec in sorted(specs, key=lambda s: s.name):
            if remainder <= 0:
                break
            allocated_counts[spec.name] += 1
            remainder -= 1

        self._allocation = allocated_counts

    def get_allocation(self) -> Dict[str, int]:
        """
        Returns the calculated allocation of images per batch specification.

        Returns:
            Dict[str, int]: A dictionary mapping specification names to the
                            number of images allocated to them.
        """
        return self._allocation

    def task_list(self) -> List[BatchSpecification]:
        """
        Generates a full, interleaved list of which specification to use for
        each image to be generated.

        Returns:
            List[BatchSpecification]: A list of BatchSpecification objects, with a
                                      length equal to total_images.
        """
        tasks: List[BatchSpecification] = []
        remaining_counts = self._allocation.copy()
        
        # Get specs sorted by name for deterministic interleaving
        sorted_specs = sorted(self.config.specifications, key=lambda s: s.name)

        while len(tasks) < self.config.total_images:
            for spec in sorted_specs:
                if remaining_counts[spec.name] > 0:
                    tasks.append(spec)
                    remaining_counts[spec.name] -= 1
        
        return tasks
