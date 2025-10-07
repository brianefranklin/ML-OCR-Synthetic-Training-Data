"""
Example of how to modify main.py to integrate FontHealthManager

This shows the specific changes needed to add adaptive font health
to the existing OCR generation pipeline.
"""

# ============================================================================
# ADD THIS IMPORT SECTION TO main.py
# ============================================================================
from font_health_manager import FontHealthManager


# ============================================================================
# MODIFY THE main() FUNCTION
# ============================================================================
def modified_main_sections():
    """
    This shows the key modifications to make in main.py
    """
    
    # 1. INITIALIZE FONT HEALTH MANAGER (add after argument parsing)
    # ------------------------------------------------------------
    """
    # --- Configure Logging ---
    setup_logging(args.log_level, args.log_file)
    logging.info("Script started.")
    
    # NEW: Initialize Font Health Manager
    font_health_manager = FontHealthManager(
        health_file=os.path.join(args.output_dir, "font_health.json"),
        min_health_threshold=30.0,
        success_increment=1.0,
        failure_decrement=10.0,
        cooldown_base_seconds=300.0,  # 5 minutes
        auto_save_interval=50
    )
    logging.info("Font health tracking enabled")
    """
    
    # 2. REPLACE can_font_render_text FUNCTION
    # -----------------------------------------
    """
    # REPLACE the existing can_font_render_text function with:
    
    @functools.lru_cache(maxsize=None)
    def can_font_render_text(font_path, text, character_set):
        font_name = os.path.basename(font_path)
        
        # Check font health first
        if font_health_manager:
            font_health_manager.register_font(font_path)
            font = font_health_manager.fonts[font_path]
            
            if font.is_in_cooldown():
                logging.debug(f"Font {font_name} is in cooldown, skipping")
                return False
            
            if font.health_score < font_health_manager.min_health_threshold:
                logging.debug(f"Font {font_name} health too low ({font.health_score:.1f}), skipping")
                return False
        
        try:
            font = ImageFont.truetype(font_path, size=24)
            for char in text:
                if char not in character_set:
                    return False
            
            # Track character coverage on success
            if font_health_manager:
                font_health_manager.fonts[font_path].character_coverage.update(character_set)
            
            return True
        except Exception as e:
            # Record failure in health manager
            if font_health_manager:
                font_health_manager.record_failure(font_path, reason="validation_error")
            logging.warning(f"Skipping font {font_name} due to error: {e}")
            return False
    """
    
    # 3. MODIFY generate_with_batches FUNCTION
    # -----------------------------------------
    """
    def generate_with_batches(batch_config, font_files, background_images, args):
        # ... existing code ...
        
        # MODIFICATION 1: Filter fonts using health manager
        healthy_fonts = font_health_manager.get_available_fonts(font_files)
        if not healthy_fonts:
            logging.error("No healthy fonts available for generation")
            return
        logging.info(f"Using {len(healthy_fonts)}/{len(font_files)} healthy fonts")
        
        # ... existing batch manager setup ...
        
        while successful_count < target_count and attempt_count < max_attempts:
            task = batch_manager.get_next_task()
            # ... existing task processing ...
            
            # MODIFICATION 2: Use health-aware font selection
            font_path = task['font_path']
            
            # Check if this specific font is healthy
            if not font_health_manager.get_available_fonts([font_path]):
                logging.debug(f"Font {os.path.basename(font_path)} is unhealthy, selecting alternative")
                # Get alternative from same batch
                batch_fonts = batch_manager._get_batch_fonts(task['batch'])
                healthy_batch_fonts = font_health_manager.get_available_fonts(batch_fonts)
                if healthy_batch_fonts:
                    font_path = font_health_manager.select_font_weighted(healthy_batch_fonts)
                else:
                    logging.warning(f"No healthy fonts for batch '{batch_name}'")
                    failed_attempts += 1
                    continue
            
            # ... existing text extraction ...
            
            try:
                # MODIFICATION 3: Track success/failure
                final_image, metadata, text, augmentations_applied = generator.generate_image(
                    # ... existing parameters ...
                )
                
                # Record success
                font_health_manager.record_success(font_path, text_line)
                
                # ... existing save logic ...
                
            except OSError as e:
                # Record specific failure types
                if "execution context too long" in str(e):
                    font_health_manager.record_failure(font_path, reason="freetype_error")
                else:
                    font_health_manager.record_failure(font_path, reason="os_error")
                failed_attempts += 1
                logging.warning(f"Font {os.path.basename(font_path)} failed: {e}")
                continue
            except Exception as e:
                # Record general failures
                font_health_manager.record_failure(font_path, reason=type(e).__name__)
                failed_attempts += 1
                logging.error(f"Failed to generate image: {e}")
                continue
        
        # MODIFICATION 4: Save health state and report
        font_health_manager.save_state()
        health_report = font_health_manager.get_summary_report()
        logging.info(f"Font health summary: {health_report}")
        
        # ... rest of function ...
    """
    
    # 4. MODIFY REGULAR GENERATION LOOP (non-batch mode)
    # ---------------------------------------------------
    """
    for i in range(args.num_images):
        # ... existing time check and text extraction ...
        
        # MODIFICATION: Health-aware font selection
        if args.font_name:
            font_path = os.path.join(args.fonts_dir, args.font_name)
            # Check if specified font is healthy
            if not font_health_manager.get_available_fonts([font_path]):
                logging.warning(f"Specified font {args.font_name} is unhealthy")
                # Could fallback to other fonts or skip
                continue
        else:
            # Select from healthy fonts only
            healthy_fonts = font_health_manager.get_available_fonts(font_files)
            if not healthy_fonts:
                logging.error("No healthy fonts available")
                break
            font_path = font_health_manager.select_font_weighted(healthy_fonts)
        
        try:
            # ... existing generation code ...
            final_image, metadata, text, augmentations_applied = generator.generate_image(
                # ... parameters ...
            )
            
            # Record success
            font_health_manager.record_success(font_path, text_line)
            
            # ... existing save code ...
            
        except Exception as e:
            # Record failure
            font_health_manager.record_failure(
                font_path, 
                reason="render_error" if "render" in str(e).lower() else "unknown"
            )
            logging.error(f"Failed to generate image: {e}")
            continue
    
    # Save final state
    font_health_manager.save_state()
    """
    
    # 5. ADD CLEANUP ON EXIT
    # -----------------------
    """
    # At the very end of main():
    
    # Print final health report
    if font_health_manager:
        report = font_health_manager.get_summary_report()
        logging.info(f"Final font health report: {report}")
        
        # List problematic fonts
        problematic = []
        for font_path, font in font_health_manager.fonts.items():
            if font.health_score < 50 or font.failure_count > font.success_count:
                problematic.append({
                    'name': os.path.basename(font_path),
                    'health': font.health_score,
                    'success_rate': font.success_count / max(1, font.success_count + font.failure_count)
                })
        
        if problematic:
            logging.warning(f"Problematic fonts detected: {problematic}")
        
        font_health_manager.save_state()
    
    logging.info("Script finished.")
    """


# ============================================================================
# OPTIONAL: Command-line tool to manage font health
# ============================================================================
def font_health_cli():
    """
    Standalone CLI tool to inspect and manage font health.
    Save this as font_health_cli.py
    """
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Font Health Management Tool')
    parser.add_argument('health_file', help='Path to font_health.json')
    parser.add_argument('--report', action='store_true', 
                       help='Show health report')
    parser.add_argument('--reset', metavar='FONT', 
                       help='Reset health for specific font')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up expired cooldowns')
    parser.add_argument('--export-blacklist', metavar='FILE',
                       help='Export unhealthy fonts to blacklist file')
    
    args = parser.parse_args()
    
    manager = FontHealthManager(health_file=args.health_file)
    
    if args.report:
        print("\n=== Font Health Report ===")
        summary = manager.get_summary_report()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        print("\n=== Individual Fonts ===")
        for font_path, font in sorted(manager.fonts.items(), 
                                     key=lambda x: x[1].health_score, 
                                     reverse=True):
            status = manager._get_font_status(font)
            print(f"{os.path.basename(font_path):30} "
                  f"Health: {font.health_score:6.1f} "
                  f"Status: {status:10} "
                  f"Success: {font.success_count:4} "
                  f"Failure: {font.failure_count:4}")
    
    if args.reset:
        manager.reset_font(args.reset)
        manager.save_state()
        print(f"Reset font: {args.reset}")
    
    if args.cleanup:
        count = manager.cleanup_stale_cooldowns()
        manager.save_state()
        print(f"Cleaned up {count} expired cooldowns")
    
    if args.export_blacklist:
        blacklist = []
        for font_path, font in manager.fonts.items():
            if font.health_score < manager.min_health_threshold:
                blacklist.append(os.path.basename(font_path))
        
        with open(args.export_blacklist, 'w') as f:
            json.dump(blacklist, f, indent=2)
        print(f"Exported {len(blacklist)} unhealthy fonts to {args.export_blacklist}")


if __name__ == "__main__":
    # Example usage
    print("This file shows how to integrate FontHealthManager into main.py")
    print("Copy the relevant sections into your main.py file")