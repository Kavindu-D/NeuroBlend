import numpy as np
import os

# Load the real PET scan
pet_path = os.path.join('NACC000314', 'pet_processed.npy')
pet = np.load(pet_path)

# Normalize to [0, 1]
pet = (pet - pet.min()) / (pet.max() - pet.min())

# Save as the PET template for blending
out_path = os.path.join('app', 'pet_template.npy')
np.save(out_path, pet.astype(np.float32))

print(f'PET template saved to {out_path}')
