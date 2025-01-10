import os
from glob import glob
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from data.pix2pix_dataset import Pix2pixDataset  # Assuming you have the original file
from data.image_folder import make_dataset


class OralDataset(Pix2pixDataset):
    """
    Adapted from Pix2pixDataset to handle a specialized folder structure for oral datasets.
    
    Directory structure (example):
        root/
          train/
            train_A/
              *.jpg
            train_A_mask/
              *.png
            train_B/
              *.jpg
            train_B_mask/
              *.png
          test/
            test_A/
              *.jpg
            test_A_mask/
              *.png
            test_B/
              *.jpg
            (No mask files for B in test mode, optional)
    
    Inherits from Pix2pixDataset to reuse transformation methods, no_pairing_check, etc.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # Use Pix2pixDataset's arguments as a base
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        
        # Override or set defaults specific to OralDataset
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=13)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(no_instance=True)

        return parser

    def initialize(self, opt):
        """
        Initialize the dataset, read in all file paths, apply sanity checks, 
        and store the final size of the dataset.
        """
        self.opt = opt
        self.mode = 'train' if opt.phase == 'train' else 'test'

        # Retrieve all relevant paths
        self.A_paths, self.A_mask_paths, self.B_paths, self.B_mask_paths = self.get_paths(opt)

        # If not skipping pairing checks, make sure everything lines up
        if not opt.no_pairing_check:
            self.verify_paths()

        # Build PyTorch transformation pipeline. 
        # You can still use get_params() / get_transform() if you want dynamic cropping, 
        # but here is a static pipeline as an example:
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

        self.dataset_size = len(self.A_paths)
        print(f"OralDataset size: {self.dataset_size}")

    def get_paths(self, opt):
        """
        Collect all file paths for A, A_mask, B, and B_mask according to the specialized 
        directory structure. Also prints out the count of each for debugging.
        """
        root = opt.dataroot
        phase = 'test' if self.mode == 'test' else 'train'

        # Collect images in each subfolder
        A_paths = sorted(glob(os.path.join(root, phase, f"{phase}_A", "*.jpg")))
        A_mask_paths = sorted(glob(os.path.join(root, phase, f"{phase}_A_mask", "*.png")))
        B_paths = sorted(glob(os.path.join(root, phase, f"{phase}_B", "*.jpg")))

        # In test mode, B_mask may not exist or is optional
        if self.mode == 'train':
            B_mask_paths = sorted(glob(os.path.join(root, phase, f"{phase}_B_mask", "*.png")))
        else:
            B_mask_paths = None

        print(f"A_paths: {len(A_paths)}")
        print(f"A_mask_paths: {len(A_mask_paths)}")
        print(f"B_paths: {len(B_paths)}")
        print(f"B_mask_paths: {len(B_mask_paths)}" if B_mask_paths else "No B_mask_paths in test mode")

        return A_paths, A_mask_paths, B_paths, B_mask_paths

    def verify_paths(self):
        """
        Additional checks to ensure A, A_mask, B, and B_mask line up as expected.
        You can optionally compare filenames as well, similar to Pix2pixDataset.paths_match().
        """
        # Basic length checks
        if self.A_mask_paths:
            assert len(self.A_paths) == len(self.A_mask_paths), \
                "Mismatch between A images and A masks lengths"

        if self.B_paths:
            assert len(self.A_paths) == len(self.B_paths), \
                "Mismatch between A and B lengths"

        # Optionally, do a more robust filename match if needed
        # for A_path, A_mask_path in zip(self.A_paths, self.A_mask_paths):
        #     assert self.paths_match(A_path, A_mask_path), \
        #         f"Filename mismatch: {A_path} vs. {A_mask_path}"

    @staticmethod
    def paths_match(path1, path2):
        """
        Check if two paths match by comparing filenames (no extension).
        Mimics the method from Pix2pixDataset.
        """
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def create_empty_mask(self):
        """Create an all-white mask if none is available."""
        return Image.fromarray(np.ones((256, 256, 3), dtype=np.uint8) * 255)

    def load_and_transform_image(self, image_path, is_mask=False):
        """
        Load and transform an image. If the path doesn't exist and `is_mask=True`,
        generate a placeholder (white) mask.
        """
        if image_path is None and is_mask:
            return self.transform(self.create_empty_mask())

        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return self.transform(img)

    def __getitem__(self, index):
        """
        Fetch a paired sample from the dataset:
          - A, A_mask: Source image and mask
          - B, B_mask: Target image and mask (B_mask may not exist in test mode)
        
        Return a dictionary consistent with the Pix2pixDataset structure:
          - 'label' (tensor)
          - 'image' (tensor)
          - 'instance' (tensor or 0)
          - plus any specialized fields (A_mask, B_mask, etc.).
        """
        # Get the relevant paths for this index
        A_path = self.A_paths[index]
        A_mask_path = self.A_mask_paths[index] if self.A_mask_paths else None
        B_path = self.B_paths[index] if self.B_paths else None
        B_mask_path = self.B_mask_paths[index] if self.B_mask_paths else None

        # Load images (convert to Tensor, apply normalization, etc.)
        A = self.load_and_transform_image(A_path)
        A_mask = self.load_and_transform_image(A_mask_path, is_mask=True)
        B = self.load_and_transform_image(B_path)
        B_mask = self.load_and_transform_image(B_mask_path, is_mask=True) \
                 if (B_mask_path is not None and self.mode == 'train') else None

        # Prepare the dictionary that Pix2pix expects
        # In many pix2pix-based datasets, 'label' is the segmentation or input,
        # and 'image' is the real target image. We'll map:
        #   A --> label
        #   B --> image
        #   A_mask / B_mask are just extras we keep for later
        input_dict = {
            'label': A,       # Use A as label
            'image': B,       # Use B as image
            'path': B_path,   # Path to the "real" image
            'A': A,
            'A_mask': A_mask,
            'B': B
        }
        if B_mask is not None:
            input_dict['B_mask'] = B_mask

        # If instance maps are not used, set them to 0
        if not self.opt.no_instance:
            instance_tensor = 0
            input_dict['instance'] = instance_tensor

        return input_dict

    def __len__(self):
        return self.dataset_size

    # -------------- Optional Utilities for Visualization -------------- #
    @staticmethod
    def tensor_to_image(tensor):
        """
        Convert a normalized [-1,1] tensor into a [0,1] range numpy array for visualization.
        """
        img = tensor.detach().cpu()
        img = img * 0.5 + 0.5  # Denormalize from [-1,1] to [0,1]
        img = img.numpy().transpose(1, 2, 0)
        return np.clip(img, 0, 1)

    def plot_sample(self, index):
        """
        Quick utility to visualize a sample with matplotlib.
        Shows A, B, and their corresponding masks if available.
        """
        import matplotlib.pyplot as plt
        sample = self[index]

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('OralDataset Sample Visualization')

        axes[0, 0].imshow(self.tensor_to_image(sample['A']))
        axes[0, 0].set_title('Input Image (A)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(self.tensor_to_image(sample['B']))
        axes[0, 1].set_title('Target Image (B)')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(self.tensor_to_image(sample['A_mask']))
        axes[1, 0].set_title('Input Mask (A_mask)')
        axes[1, 0].axis('off')

        if 'B_mask' in sample:
            axes[1, 1].imshow(self.tensor_to_image(sample['B_mask']))
            axes[1, 1].set_title('Target Mask (B_mask)')
        axes[1, 1].axis('off')

        plt.show()
