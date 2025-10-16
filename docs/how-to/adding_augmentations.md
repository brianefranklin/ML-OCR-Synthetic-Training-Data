# How-To: Add a New Augmentation

This guide provides a step-by-step recipe for adding a new effect or augmentation to the data generation pipeline, following the project's Test-Driven Development (TDD) workflow.

Let's assume we want to add a new effect called "invert".

## 1. Write a Failing Test (Red)

First, create a new test in the appropriate test file. Since "invert" is a simple image-level effect, we'll add it to `tests/test_effects.py`.

- **Create a new test function:** `test_invert_is_applied`.
- **Import the (not yet existing) function:** `from src.effects import apply_invert`.
- **Write the test:** The test should create a simple image, call the new function, and assert that the output is different from the input. This test *must fail* initially.

```python
# in tests/test_effects.py

# ... add apply_invert to the import list

def test_invert_is_applied():
    """Tests that the invert effect modifies the image."""
    # Create a simple, non-symmetrical image
    image = Image.new("RGB", (100, 50), "white")
    draw = ImageDraw.Draw(image)
    draw.line((0, 0, 100, 50), fill="black")
    original_array = np.array(image)

    # Call the function that doesn't exist yet
    inverted_image = apply_invert(image)
    inverted_array = np.array(inverted_image)

    # Assert that the image has changed
    assert not np.array_equal(original_array, inverted_array)
```

Running `pytest` at this point will result in an `ImportError`.

## 2. Implement the Function (Green)

Now, implement the simplest possible version of the function in the corresponding source file (`src/effects.py`) to make the test pass.

```python
# in src/effects.py
from PIL import ImageOps

# ... other functions

def apply_invert(image: Image.Image) -> Image.Image:
    """Inverts the colors of an image."""
    return ImageOps.invert(image.convert("RGB"))
```

Running `pytest tests/test_effects.py` should now result in all tests passing.

## 3. Integrate into the Generator

Finally, wire the new function into the main `OCRDataGenerator` pipeline.

1.  **Update `plan_generation`:** In `src/generator.py`, add a new parameter to the `plan_generation` method signature (e.g., `invert: bool = False`) and add it to the dictionary it returns.

    ```python
    # in OCRDataGenerator.plan_generation(...)
    return {
        # ... other params
        "invert": invert,
    }
    ```

2.  **Update `generate_from_plan`:** In the `generate_from_plan` method, get the new parameter from the plan and call your function.

    ```python
    # in OCRDataGenerator.generate_from_plan(...)
    
    # ... after other augmentations
    if plan.get("invert"):
        final_image = apply_invert(final_image)
    
    return final_image, final_bboxes
    ```

3.  **Add an Integration Test:** Add a final test to `tests/test_generator.py` that calls the public `plan_generation` and `generate_from_plan` methods and confirms that the new effect is applied correctly from end to end.

This TDD process ensures that every new feature is testable, isolated, and correctly integrated into the main application.
