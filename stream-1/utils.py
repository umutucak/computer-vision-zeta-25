from ast import literal_eval
import os
import matplotlib.pyplot as plt
from skimage import io , img_as_uint 
import pandas as pd
import matplotlib.patches as mpatches


def process_labels(labels_dir,split):
    path = os.path.join(labels_dir, f"{split}.csv")
    labels = pd.read_csv(path)
    return  labels



class SPARKDataset:

    """ Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data. """

    def __init__(self, class_map, root_dir='./data',split='train'):
        self.root_dir = os.path.join(root_dir)
        self.labels = process_labels(root_dir,split)
        self.class_map =  class_map
        self.split = split

    def get_sample(self, i=0):

        """ Loading image as PIL image. """

        sat_name = self.labels.iloc[i]['Class']
        img_name = self.labels.iloc[i]['Image name']
        mask_name = self.labels.iloc[i]['Mask name']
        bbox = self.labels.iloc[i]['Bounding box']

        image_file = f'{self.root_dir}/images/{sat_name}/{self.split}/{img_name}'
        mask_file = f'{self.root_dir}/mask/{sat_name}/{self.split}/{mask_name}'

        bbox = literal_eval(bbox)
        image = io.imread(image_file)
        mask = io.imread(mask_file)


        return {"img": image , "mask":mask, "bbox":bbox, "class":self.class_map[sat_name]}


    def visualize(self, i, size=(15, 15), ax=None, mask_visualize=False):
        """Visualizing image, with ground truth pose and optional mask overlay."""

        if ax is None:
            ax = plt.gca()
            
        sample = self.get_sample(i)
        image = sample['img']
        mask = sample['mask']
        img_class = sample['class']
        bbox = sample['bbox']
        x_min, y_min, x_max, y_max = bbox

        if mask_visualize:
            ax.imshow(image, vmin=0, vmax=255)
            ax.imshow(mask, alpha=0.5)  # Add overlay with semi-transparent mask

        else:
            ax.imshow(image, vmin=0, vmax=255)

        # Adjust rectangle creation to match the new bbox format (Pascal VOC format)
        rect = mpatches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        label = f"{list(self.class_map.keys())[list(self.class_map.values()).index(img_class)]}"

        # Adjust text placement to reflect the new coordinate system
        ax.text(x_min, y_min-50, label, color='white', fontsize=10, verticalalignment='top')

        ax.set_axis_off()



    
try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    has_pytorch = True
    print('Found Pytorch')
except ImportError:
    has_pytorch = False

    
if has_pytorch:
    
    class PyTorchSPARKDataset(Dataset):
        """PyTorch Dataset class for SPARK dataset."""

        def __init__(self, class_map, root_dir='./data', split='train', transform=None):
            self.root_dir = os.path.join(root_dir)
            self.labels = process_labels(root_dir, split)
            self.class_map = class_map
            self.split = split
            self.transform = transform

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            sat_name = self.labels.iloc[idx]['Class']
            img_name = self.labels.iloc[idx]['Image name']
            mask_name = self.labels.iloc[idx]['Mask name']
            bbox = literal_eval(self.labels.iloc[idx]['Bounding box'])

            image_file = f'{self.root_dir}/images/{sat_name}/{self.split}/{img_name}'
            mask_file = f'{self.root_dir}/mask/{sat_name}/{self.split}/{mask_name}'

            image = io.imread(image_file)
            mask = io.imread(mask_file)

            image = transforms.ToTensor()(image)
            mask  = torch.tensor(mask, dtype=torch.uint8)[None]

            x_min, y_min, x_max, y_max = bbox
            cls_id = self.class_map[sat_name]

            target = {
                "boxes": torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32),
                "labels": torch.tensor([cls_id], dtype=torch.int64),
                "masks": mask,
                "image_id": torch.tensor([idx]),
                "area": torch.tensor([(x_max - x_min) * (y_max - y_min)], dtype=torch.float32),
                "iscrowd": torch.tensor([0], dtype=torch.int64),
                "class": torch.tensor(self.class_map[sat_name], dtype=torch.long)
            }

            return image, target
else:
    class PyTorchSparkDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError('Pytorch is not available!')
            