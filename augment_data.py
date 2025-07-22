import csv
import random

# Read original data
original_data = []
with open('data/fem_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        original_data.append({k: float(v) for k, v in row.items()})

# Generate augmented data
augmented_data = []
headers = list(original_data[0].keys())
for _ in range(1000):
    # Pick a random base entry
    base_entry = random.choice(original_data).copy()
    
    # Add noise
    # Noise is proportional to the value, e.g., up to +/- 2%
    for key in headers:
        # For frequency, noise is smaller to keep it close to original classes
        if key == 'frequency':
            noise_factor = 1 + (random.random() - 0.5) * 0.01 # +/- 0.5%
        else:
            noise_factor = 1 + (random.random() - 0.5) * 0.04 # +/- 2%
        base_entry[key] *= noise_factor
    
    augmented_data.append(base_entry)

# Write new CSV
with open('data/fem_data_augmented.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(augmented_data)

print('Successfully created data/fem_data_augmented.csv with 1000 entries.')
