import numpy as np
import cv2
import os

output_dir = './training_data/X'
os.makedirs(output_dir, exist_ok=True)

width, height = 28, 28
num_images = 800
min_x_size = 3

for i in range(num_images):
    image = np.zeros((height, width), dtype=np.uint8)
    x_size = np.random.randint(min_x_size, min(width, height) // 2)
    angle = np.random.randint(0, 360)
    thickness = np.random.randint(1, 5)
    # Random center position for the X ensuring 80% of the X is still in the image
    margin = int(x_size * 0.8)
    center_x = np.random.randint(margin, width - margin)
    center_y = np.random.randint(margin, height - margin)
    
    x_image = np.zeros((height, width), dtype=np.uint8)
    cv2.line(x_image, (center_x - x_size, center_y - x_size), (center_x + x_size, center_y + x_size), 255, thickness)
    cv2.line(x_image, (center_x + x_size, center_y - x_size), (center_x - x_size, center_y + x_size), 255, thickness)
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    x_image = cv2.warpAffine(x_image, M, (width, height))
    image = cv2.add(image, x_image)
    image_path = os.path.join(output_dir, f'image_{i}.png')
    cv2.imwrite(image_path, image)

print(f'{num_images} images have been generated and saved to {output_dir}')
