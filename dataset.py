import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw
import random
import os
import json
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MNISTBlackBackgroundDataset:
    def __init__(self, 
                 root: str = './mnist_data_2',
                 train: bool = True,
                 image_size: int = 352,
                 max_digits_per_image: int = 5,
                 min_digits_per_image: int = 1,
                 transform=None):
        """
        MNIST Object Detection Dataset with Black Background
        
        Args:
            root: Root directory for data storage
            train: Whether to use training or test set
            image_size: Size of output images (square)
            max_digits_per_image: Maximum digits per image
            min_digits_per_image: Minimum digits per image
            transform: Optional transforms to apply
        """
        self.root = root
        self.train = train
        self.image_size = image_size
        self.max_digits_per_image = max_digits_per_image
        self.min_digits_per_image = min_digits_per_image
        self.transform = transform
        
        # Load MNIST dataset
        self.mnist = torchvision.datasets.MNIST(
            root=root, 
            train=train, 
            download=True,
            transform=transforms.ToTensor()
        )
        
        # Create output directories
        os.makedirs(os.path.join(root, 'images'), exist_ok=True)
        os.makedirs(os.path.join(root, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(root, 'annotations'), exist_ok=True)
        
    def create_dataset(self, num_images: int = 1000):
        """Create the object detection dataset with black background"""
        annotations = {
            "info": {
                "description": "MNIST Object Detection Dataset with Black Background",
                "version": "1.0",
                "year": 2024,
                "contributor": "Generated from MNIST"
            },
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i, "name": str(i), "supercategory": "digit"} for i in range(10)
            ]
        }
        
        annotation_id = 1

        #creatre train.txt and val.txt
        trainpath = []
        valpath = []
        
        for img_idx in range(num_images):
            # Create black background image
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            
            # Random number of digits
            num_digits = random.randint(self.min_digits_per_image, self.max_digits_per_image)
            
            image_annotations = []
            
            for _ in range(num_digits):
                # Get random MNIST digit
                idx = random.randint(0, len(self.mnist) - 1)
                digit_img, digit_label = self.mnist[idx]
                
                # Convert MNIST tensor to numpy array and remove background
                digit_np = (digit_img.squeeze().numpy() * 255).astype(np.uint8)
                
                # Find where the digit is (non-zero pixels)
                y_indices, x_indices = np.where(digit_np > 10)  # Threshold to find digit
                
                if len(x_indices) == 0 or len(y_indices) == 0:
                    continue
                    
                # Get digit bounding box
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                digit_height = y_max - y_min + 1
                digit_width = x_max - x_min + 1
                
                # Extract the digit from the image
                digit_cropped = digit_np[y_min:y_max+1, x_min:x_max+1]
                
                # Random scale for the digit
                scale = random.uniform(0.8, 1.5)
                new_height = max(10, min(80, int(digit_height * scale)))
                new_width = max(10, min(80, int(digit_width * scale)))
                
                # Resize digit
                if new_height > 0 and new_width > 0:
                    # Use OpenCV for resizing if available, otherwise use PIL
                    try:
                        import cv2
                        digit_resized = cv2.resize(digit_cropped, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    except ImportError:
                        digit_pil = Image.fromarray(digit_cropped)
                        digit_resized = np.array(digit_pil.resize((new_width, new_height), Image.BILINEAR))
                
                # Random position (ensure digit fits within image)
                max_x = self.image_size - new_width
                max_y = self.image_size - new_height
                
                if max_x <= 0 or max_y <= 0:
                    continue
                    
                x_pos = random.randint(0, max_x)
                y_pos = random.randint(0, max_y)
                
                # Place digit on black background
                for i in range(new_height):
                    for j in range(new_width):
                        if y_pos + i < self.image_size and x_pos + j < self.image_size:
                            pixel_value = digit_resized[i, j]
                            if pixel_value > 10: # Only copy non-background pixels
                                image[y_pos + i, x_pos + j] = [pixel_value, pixel_value, pixel_value]
                
                # Create bounding box annotation (YOLO format: center_x, center_y, width, height normalized)
                center_x = (x_pos + new_width / 2) / self.image_size
                center_y = (y_pos + new_height / 2) / self.image_size
                width_norm = new_width / self.image_size
                height_norm = new_height / self.image_size
                
                image_annotations.append({
                    'class_id': digit_label,
                    'bbox': [center_x, center_y, width_norm, height_norm]
                })
                
                # Add to COCO annotations
                coco_bbox = [x_pos, y_pos, new_width, new_height]
                annotations["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_idx,
                    "category_id": int(digit_label),
                    "bbox": coco_bbox,
                    "area": new_width * new_height,
                    "iscrowd": 0
                })
                annotation_id += 1
            
            # Save image
            img_filename = f"{'train' if self.train else 'val'}_{img_idx:06d}.jpg"
            img_path = os.path.join(self.root, 'images', img_filename)
            outerroot = os.path.abspath(self.root)
            if self.train:
                trainpath.append(os.path.join(outerroot, 'images', img_filename))
            else:
                valpath.append(os.path.join(outerroot, 'images', img_filename))

            # Convert to PIL Image and save
            pil_image = Image.fromarray(image)
            pil_image.save(img_path)
            
            # Save YOLO format labels
            label_filename = f"{'train' if self.train else 'val'}_{img_idx:06d}.txt"
            label_path = os.path.join(self.root, 'labels', label_filename)
            
            with open(label_path, 'w') as f:
                for ann in image_annotations:
                    f.write(f"{ann['class_id']} {ann['bbox'][0]:.6f} {ann['bbox'][1]:.6f} {ann['bbox'][2]:.6f} {ann['bbox'][3]:.6f}\n")
            
            # Add image info to COCO annotations
            annotations["images"].append({
                "id": img_idx,
                "file_name": img_filename,
                "width": self.image_size,
                "height": self.image_size
            })
            
            if (img_idx + 1) % 100 == 0:
                print(f"Created {img_idx + 1}/{num_images} images")
        

        
        # Create train.txt and val.txt
        if self.train:
            with open(os.path.join(self.root, 'train.txt'), 'w') as f:
                f.writelines([f"{p}\n" for p in trainpath])
        else:
            with open(os.path.join(self.root,  'val.txt'), 'w') as f:
                f.writelines([f"{p}\n" for p in valpath])
        # Save COCO format annotations
        coco_path = os.path.join(self.root, 'annotations', 
                               f"instances_{'train' if self.train else 'val'}.json")
        with open(coco_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"Created dataset with {num_images} images in {self.root}")

    def visualize_sample(self, num_samples: int = 5):
        """Visualize random samples from the dataset with bounding boxes"""
        # Find available images
        image_files = [f for f in os.listdir(os.path.join(self.root, 'images')) 
                      if f.startswith('train_' if self.train else 'val_')]
        
        if not image_files:
            print("No images found. Please create the dataset first.")
            return
            
        fig, axes = plt.subplots(1, min(num_samples, 5), figsize=(15, 5))
        if num_samples == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            if i >= len(image_files):
                break
                
            # Load image
            img_path = os.path.join(self.root, 'images', image_files[i])
            image = Image.open(img_path)
            ax.imshow(np.array(image), cmap='gray')
            
            # Load corresponding labels
            label_path = os.path.join(self.root, 'labels', image_files[i].replace('.jpg', '.txt'))
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, w, h = map(float, parts)
                        
                        # Convert YOLO format to image coordinates
                        img_w, img_h = image.size
                        x_center = cx * img_w
                        y_center = cy * img_h
                        width = w * img_w
                        height = h * img_h
                        
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        
                        # Create rectangle patch
                        rect = patches.Rectangle(
                            (x1, y1), width, height, 
                            linewidth=2, edgecolor='r', facecolor='none'
                        )
                        ax.add_patch(rect)
                        
                        # Add class label
                        ax.text(x1, y1 - 5, str(int(class_id)), color='red', 
                               fontsize=12, weight='bold', 
                               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            ax.set_title(f'Sample {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

def create_yolo_config(dataset_path: str, num_classes: int = 10):
    """Create YOLO configuration files"""
    # Create data.yaml
    data_config = {
        'train': '../mnist_black_bg/images',
        'val': '../mnist_black_bg/images',
        'nc': num_classes,
        'names': [str(i) for i in range(num_classes)]
    }
    
    with open(os.path.join(dataset_path, 'data.yaml'), 'w') as f:
        f.write(f"train: {data_config['train']}\n")
        f.write(f"val: {data_config['val']}\n")
        f.write(f"nc: {data_config['nc']}\n")
        f.write(f"names: {data_config['names']}\n")

# Example usage
if __name__ == "__main__":
    # Create training dataset
    train_dataset = MNISTBlackBackgroundDataset(
        root='mnist_data_2',
        train=True,
        image_size=352,
        max_digits_per_image=5,
        min_digits_per_image=1
    )
    
    # Create validation dataset
    val_dataset = MNISTBlackBackgroundDataset(
        root='mnist_data_2',
        train=False,
        image_size=352,
        max_digits_per_image=3,
        min_digits_per_image=1
    )
    
    # Generate datasets
    print("Creating training dataset...")
    train_dataset.create_dataset(num_images=5000)  # 100 training images
    
    print("Creating validation dataset...")
    val_dataset.create_dataset(num_images=1000)     # 20 validation images

    # Create YOLO config
    create_yolo_config('./mnist_black_bg', num_classes=10)
    
    # Visualize samples
    print("Visualizing samples...")
    train_dataset.visualize_sample(num_samples=3)
    
    print("Dataset creation complete!")
    print("Dataset structure:")
    print("mnist_black_bg/")
    print("├── images/")
    print("├── labels/")
    print("├── annotations/")
    print("└── data.yaml")