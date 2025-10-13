# Conceptual: System Architecture

The data generator is built on a two-stage, **plan-then-execute** architecture to ensure determinism, reproducibility, and flexibility.

## 1. The "Plan" Stage

- **Method:** `OCRDataGenerator.plan_generation(spec, ...)`
- **Purpose:** To decide *what* to generate, not *how*.
- **Process:** This method takes a high-level `BatchSpecification` and other inputs (like the current text and font) and produces a complete "plan" dictionary. This plan is the ground truth for a single generated image.
- **Output:** A dictionary containing every parameter required to create the image, including:
    - The source text.
    - The font file path.
    - All text direction and color information.
    - Specific values for all text effects (e.g., `ink_bleed_radius`).
    - Specific values for all image augmentations (e.g., `rotation_angle`).
    - A master `seed` for the random number generator to ensure the generation is perfectly reproducible.

This plan dictionary is what is ultimately saved as the JSON label file for each image.

## 2. The "Execute" Stage

- **Method:** `OCRDataGenerator.generate_from_plan(plan)`
- **Purpose:** To deterministically execute a generation plan.
- **Process:**
    1.  Takes a `plan` dictionary as its only input.
    2.  **Seeds the random number generator** with the `seed` from the plan. This is the most critical step for ensuring reproducibility.
    3.  Calls the internal rendering methods (`_render_text`, etc.) with the exact parameters specified in the plan.
    4.  Applies all text effects and image augmentations in a fixed order, using the parameters from the plan.
    5.  Returns the final, fully augmented image and the corresponding accurate bounding boxes.

## Benefits of this Architecture

- **Reproducibility:** Any generated image can be perfectly recreated, pixel-for-pixel, by re-running `generate_from_plan` with the image's corresponding JSON label file. This is invaluable for debugging.
- **Flexibility:** The planning stage can be made as simple or as complex as needed. It can select random parameters from wide ranges or be constrained to very specific values, without having to change the underlying generation logic.
- **Clarity:** It cleanly separates the *what* from the *how*, making the codebase easier to understand and maintain.
