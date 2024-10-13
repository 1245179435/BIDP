from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
class CustomDataset(Dataset):
    def __init__(self, img_dir, seg_dir,bian,transform=None):
        """
        img_dir: Path to the image directory.
        seg_dir: Path to the segmentation directory.
        transform: Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)
        self.bian=bian

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        seg_path = os.path.join(self.seg_dir, img_name[:-4] + ".png")  # Assuming the segmentation files are PNGs
        bian= os.path.join(self.bian, img_name[:-4] + "_edge.png")

        img = Image.open(img_path)
        seg = Image.open(seg_path)
        bian=Image.open(bian)

        sample = {'image': img, 'segmentation': seg, "filename": img_name,"bian":bian}

        if self.transform:
            sample = self.transform(sample)

        return sample

