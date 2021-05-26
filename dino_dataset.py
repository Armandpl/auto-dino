import os
import glob

import cv2
import PIL
import torch
from torchvision import transforms

class DinoDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        super(DinoDataset, self).__init__()
        self.directory = directory
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        self.refresh()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = cv2.imread(ann["image_path"], cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, ann["action"]

    def _parse(self, path):
        basename = os.path.basename(path)
        items = basename.split("_")
        action = int(items[0])

        return action

    def refresh(self):
        self.annotations = []
        for image_path in glob.glob(os.path.join(self.directory, "*.jpg")):
            action = self._parse(image_path)
            self.annotations += [{"image_path": image_path, "action": action}]
