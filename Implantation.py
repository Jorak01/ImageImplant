import cv2
import numpy as np
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch
from PIL import Image


def remove_background(image_path, output_path):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image)

    # Load the pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Perform inference
    with torch.no_grad():
        predictions = model([image_tensor])[0]

    # Get masks and scores
    masks = predictions['masks']
    scores = predictions['scores']

    # Consider only masks with a score higher than 0.5
    masks = masks[scores > 0.5]

    # If no masks found, return the original image
    if len(masks) == 0:
        image.save(output_path)
        return

    # Combine masks
    combined_mask = masks.sum(dim=0).squeeze().cpu().numpy()
    combined_mask = (combined_mask > 0.5).astype(np.uint8) * 255

    # Convert to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    mask_cv = combined_mask

    # Invert mask
    inv_mask = cv2.bitwise_not(mask_cv)

    # Apply the mask to the image
    fg = cv2.bitwise_and(image_cv, image_cv, mask=mask_cv)
    bg = np.full_like(image_cv, (0, 0, 0))

    # Save the foreground image with transparency
    fg = cv2.cvtColor(fg, cv2.COLOR_BGR2BGRA)
    fg[:, :, 3] = mask_cv
    cv2.imwrite(output_path, fg)


def add_new_background(foreground_path, background_path, output_path):
    # Open the images
    foreground = Image.open(foreground_path).convert("RGBA")
    background = Image.open(background_path).convert("RGBA")

    # Resize background to match foreground size
    background = background.resize(foreground.size)

    # Composite the images
    composite = Image.alpha_composite(background, foreground)

    # Save the final image
    composite.save(output_path)


# Example usage
input_image_path = 'input.png'
foreground_path = 'individualimage.png'
background_image_path = 'backgroundreplacement.png'
final_output_path = 'output.png'

# Remove the background
remove_background(input_image_path, foreground_path)

# Replace with a new background
add_new_background(foreground_path, background_image_path, final_output_path)

