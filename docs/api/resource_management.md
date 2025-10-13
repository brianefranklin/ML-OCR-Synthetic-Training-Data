# API Reference: Resource Management

This section describes the classes responsible for managing resources like fonts and backgrounds.

## `resource_manager.py`

### `ResourceManager`
- **Description:** A generic, reusable class that provides a score-based health tracking system for any type of resource.
- **Key Features:**
    - **Heuristic Scoring:** Maintains a "health score" for each resource. Successes increase the score, failures decrease it.
    - **Filtering:** Can provide a list of only "healthy" resources above a certain score threshold.
    - **Weighted Selection:** Selects a resource from a list with a probability weighted by its health score, making healthier resources more likely to be chosen.

## `font_health_manager.py`

### `FontHealthManager`
- **Description:** A specialized subclass of `ResourceManager` specifically for managing font files.

## `background_manager.py`

### `BackgroundImageManager`
- **Description:** A specialized subclass of `ResourceManager` for managing background images.
- **Key Features:** In addition to the `ResourceManager` features, it also includes logic to discover image files from a given list of directories.
