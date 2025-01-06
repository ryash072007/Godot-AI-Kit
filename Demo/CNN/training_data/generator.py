import numpy as np
import cv2
import os

# Directory to save the images
output_dir = '.'
os.makedirs(output_dir, exist_ok=True)

# Image dimensions
width, height = 14, 14

# Number of images to generate
num_images = 100

for i in range(num_images):
    # Create a blank grayscale image
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Randomly choose a column to place the vertical line
    line_col = np.random.randint(0, width)
    
    # Draw the vertical line
    image[:, line_col] = 255
    
    # Save the image
    image_path = os.path.join(output_dir, f'image_{i}.png')
    cv2.imwrite(image_path, image)

print(f'{num_images} images have been generated and saved to {output_dir}')