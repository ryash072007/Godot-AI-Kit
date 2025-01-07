import numpy as np
import cv2
import os

output_dir = './training_data/O'
os.makedirs(output_dir, exist_ok=True)

width, height = 28, 28
num_images = 800
min_o_size = 3
max_o_size = min(width, height) // 2
min_thickness = 1
max_thickness = 3

for i in range(num_images):
    image = np.zeros((height, width), dtype=np.uint8)
    o_size = np.random.randint(min_o_size, max_o_size)
    angle = np.random.randint(0, 360)
    thickness = np.random.randint(min_thickness, max_thickness)
    o_image = np.zeros((height, width), dtype=np.uint8)
    
    # Ensure 80% of the 'O' is still visible
    max_offset_x = width - o_size * 2
    max_offset_y = height - o_size * 2
    offset_x = np.random.randint(0, max_offset_x)
    offset_y = np.random.randint(0, max_offset_y)
    
    center_x = offset_x + o_size
    center_y = offset_y + o_size
    
    cv2.circle(o_image, (center_x, center_y), o_size, 255, thickness)
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    o_image = cv2.warpAffine(o_image, M, (width, height))
    image = cv2.add(image, o_image)
    image_path = os.path.join(output_dir, f'image_{i}.png')
    cv2.imwrite(image_path, image)

print(f'{num_images} images have been generated and saved to {output_dir}')
