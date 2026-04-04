import numpy as np
import os

# Create a random PET template (replace with real PET data if available)
pet_template = np.random.rand(160, 192, 160).astype(np.float32)

# Ensure the app directory exists
app_dir = os.path.join(os.path.dirname(__file__), 'app')
os.makedirs(app_dir, exist_ok=True)

# Save the template in the correct location
np.save(os.path.join(app_dir, 'pet_template.npy'), pet_template)

print('PET template saved to app/pet_template.npy')
