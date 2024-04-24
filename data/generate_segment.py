import os
import cv2
import numpy as np

trimaps_folder = 'trimaps'
trimaps_files = os.listdir(trimaps_folder)
trimaps_files.sort()
trimaps_files.remove('.DS_Store')
jpg_folder = 'flower'
jpg_files = os.listdir(jpg_folder)
jpg_files.sort()

output_folder = 'flowers'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the scaling factor for perturbation intensity

for jpg_file in jpg_files:
    # Generate the file paths for the original image and the segmentation mask
    jpg_path = os.path.join(jpg_folder, jpg_file)
    mask_file = os.path.splitext(jpg_file)[0] + '.png'
    mask_path = os.path.join(trimaps_folder, mask_file)
    
    perturbation_scale = 0.9

    # Load the original image
    original_image = cv2.imread(jpg_path)

    # Load the segmentation mask if it exists
    if os.path.exists(mask_path):
        segmentation_mask = cv2.imread(mask_path)

        # Resize the mask to match the image dimensions
        mask = cv2.resize(segmentation_mask, (original_image.shape[1], original_image.shape[0]))

        # Convert the segmentation mask to a binary mask
        foreground_colors = [(0, 0, 128), (0, 0, 0)]

        # Convert the segmentation mask to a binary mask
        mask = np.any([np.all(mask == color, axis=2) for color in foreground_colors], axis=0).astype(np.uint8)

        # Generate perturbation for the background
        perturbation = np.random.randint(0, 256, original_image.shape, dtype=np.uint8)
        perturbation = perturbation_scale * perturbation

        # Apply the perturbation to the background based on the mask
        perturbed_image = np.where(mask[..., np.newaxis], original_image, original_image + perturbation)
        masked_image1 = perturbed_image
        
        segmentation_mask1 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(segmentation_mask1, (original_image.shape[1], original_image.shape[0]))
        binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        masked_image = cv2.bitwise_and(masked_image1, masked_image1, mask=binary_mask)
        
    else:
        masked_image = original_image

    # Save the masked image
    output_path = os.path.join(output_folder, jpg_file)
    if masked_image is not None:
        cv2.imwrite(output_path, masked_image)
