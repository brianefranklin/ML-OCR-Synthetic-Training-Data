Implementing curved text is a significant geometric challenge that builds on our
  existing rendering pipeline. Here is a detailed, step-by-step plan for how I would approach it
  using Test-Driven Development, without writing the code itself.

  High-Level Strategy

  The core idea is to calculate the position and rotation angle for each individual character along
  a mathematical curve (an arc or a sine wave). Instead of drawing the character directly onto the
  main text surface, we will:
   1. Draw each character onto a small, temporary, transparent surface.
   2. Rotate this small surface to the correct angle for its position on the curve.
   3. Calculate the precise (x, y) coordinate on the main text surface where this rotated character
      should be pasted.
   4. Paste the rotated character onto the main surface.
   5. Calculate the final, tight bounding box of the pasted, rotated character.

  This process is repeated for every character in the string.

  ---

  Step-by-Step TDD Plan

  Phase 1: Implement Arc Text

   1. Write Failing Test for Arc Rendering (Red):
       * Create a new test file: tests/test_curved_text.py.
       * Add a test test_render_arc_text_is_applied.
       * This test will call a new, non-existent method _render_arc_text in OCRDataGenerator.
       * It will pass parameters for a simple arc (e.g., radius, start angle).
       * The test will assert that the output image is not the same as a straight-line rendering of
         the same text, proving a different rendering path was taken. This will fail with an
         AttributeError.

   2. Implement Minimal `_render_arc_text` (Green):
       * Create the _render_arc_text method in src/generator.py.
       * For now, it can just call the existing _render_text_surface to make the test fail for the
         right reason (images being identical).
       * Update the main _render_text dispatcher to call this new method when direction is, for
         example, left_to_right_arc.

   3. Implement Core Arc Logic (Green):
       * Inside _render_arc_text, implement the core algorithm:
           * Calculate the total arc length needed to fit the text.
           * Loop through each character in the input string.
           * For each character, calculate its (x, y) position on the circumference of the arc. This
             involves basic trigonometry (cos for x, sin for y).
           * Calculate the character's rotation angle, which is the tangent to the arc at that point.
           * Use the "draw-then-paste" method:
               * Draw the single character onto a small, transparent temporary PIL.Image.
               * Rotate this temporary image by the calculated angle.
               * Paste the rotated character image onto the main text surface at the calculated (x, 
                 y) position.

   4. Write Failing Test for Arc Bounding Boxes (Red):
       * Add a new test, test_arc_text_bounding_boxes_are_accurate.
       * This test will render a simple word on an arc.
       * It will assert that the bounding box for the first character is in a different position and
         likely has different dimensions than the bounding box for the last character (due to the
         rotation and placement). This will fail because we are not yet calculating the new bounding
         boxes.

   5. Implement Arc Bounding Box Calculation (Green):
       * To calculate the bounding box of a rotated character, we can't just rotate the original box.
       * After rotating the temporary single-character surface, we will get its new width and height.
       * The final bounding box will be (paste_x, paste_y, paste_x + new_width, paste_y + 
         new_height).
       * This ensures a tight, accurate, axis-aligned bounding box for each rotated character.
    
    6. Update all documentation
       * Review in-code comments and docstrings to ensure they are factually correct and meet project standards
       * Update relevant documentation in ./doc/ updating existing and creating new as appropriate

  Phase 2: Implement Sine Wave Text

  This phase will be very similar to the arc implementation, just with a different mathematical
  function.

   1. Write Failing Test for Sine Wave (Red):
       * Add a test test_render_sine_wave_text_is_applied that calls a new _render_sine_wave_text
         method.
       * Assert that the output is different from a straight-line rendering.

   2. Implement Sine Wave Logic (Green):
       * In the new _render_sine_wave_text method:
           * Loop through each character.
           * Calculate its x position by advancing horizontally as normal.
           * Calculate its y position using a sine function: y = amplitude * sin(frequency * x).
           * Calculate the character's rotation angle by finding the derivative of the sine wave at
             that point (atan(amplitude * frequency * cos(frequency * x))).
           * Use the same "draw-rotate-paste" technique as the arc renderer.

   3. Verify Bounding Boxes:
       * The bounding box calculation logic from the arc implementation will be reused here, as it's
         fundamentally the same process. A new test, test_sine_wave_bounding_boxes_are_accurate, will
         be added to confirm this.

    4. Update all documentation
       * Review in-code comments and docstrings to ensure they are factually correct and meet project standards
       * Update relevant documentation in ./doc/ updating existing and creating new as appropriate
       
  Phase 3: Refactor and Integrate

   1. Create a `_render_curved_text` Dispatcher: To avoid code duplication between the arc and sine
      wave renderers, I would create a single, generic _render_curved_text method. This method would
      take a "curve function" as an argument, which would be responsible for providing the (x, y, 
      angle) for each character. The _render_arc_text and _render_sine_wave_text methods would then
      become simple wrappers that call this generic method with the correct curve function.
   2. Integrate with `plan_generation`: Update the BatchSpecification and plan_generation to support
      new direction options (e.g., left_to_right_arc, top_to_bottom_sine) and parameters for the
      curves (e.g., arc_radius, sine_wave_amplitude).
  3. Update all documentation
      * Review in-code comments and docstrings to ensure they are factually correct and meet project standards
      * Update relevant documentation in ./doc/ updating existing and creating new as appropriate
       
  This step-by-step, test-driven plan would allow us to build this complex feature reliably, ensuring
   that both the visual output and the critical bounding box data are correct at every stage.

