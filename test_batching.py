import yaml

# Load the YAML file
# Assuming a file named 'batch_configs/comprehensive_ltr_all_features.yaml' exists
# For demonstration, let's create a dummy config if the file doesn't exist.
try:
    with open('batch_configs/comprehensive_ltr_all_features.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("⚠️  Warning: 'batch_configs/comprehensive_ltr_all_features.yaml' not found.")
    print("Using a dummy configuration for demonstration purposes.")
    config = {
        'total_images': 1000,
        'batches': [
            {'name': 'batch_a', 'proportion': 0.3, 'background_dirs': ['/path/to/bg1']},
            {'name': 'batch_b', 'proportion': 0.5},
            {'name': 'batch_c', 'proportion': 0.2, 'background_dirs': ['/path/to/bg2']},
        ]
    }


# Calculate total proportion
total_proportion = sum(batch['proportion'] for batch in config['batches'])
total_images = config['total_images']
batch_count = len(config['batches'])

print(f'✅ Configuration Summary')
print(f'=' * 50)
print(f'Total images: {total_images}')
print(f'Number of batches: {batch_count}')
print(f'Total proportion: {total_proportion:.6f}')
print(f'Difference from 1.0: {abs(total_proportion - 1.0):.9f}')

# Count batches with background_dirs
bg_batches = [b for b in config['batches'] if 'background_dirs' in b]
print(f'\n📸 Background Image Batches')
print(f'=' * 50)
print(f'Batches with background images: {len(bg_batches)}')
total_bg_images = 0
for batch in bg_batches:
    img_count = int(batch['proportion'] * total_images)
    total_bg_images += img_count
    # Correctly access the "name" key
    print(f'  • {batch["name"]}: {img_count} images')

# Corrected the newline character from '\\n' to '\n'
print(f'\nTotal background images: {total_bg_images} ({100*total_bg_images/total_images:.1f}%)')

# FIX: The original script had an unterminated f-string because it was split
# across two lines. It has been combined into a single line here.
print(f'Total non-background images: {total_images - total_bg_images} ({100*(total_images-total_bg_images)/total_images:.1f}%)')
