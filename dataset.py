import glob
import os
from PIL import Image


def default_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class Dataset:

    def __init__(self, image_root_path, data_transforms=None):
        self.data_transforms = data_transforms
        self.image_root_path = image_root_path
        self.images = []
        self.labels = []
        classes_folders = os.listdir(self.image_root_path)
        for cls_folder in classes_folders:
            folder_path = os.path.join(self.image_root_path, cls_folder)
            if os.path.isdir(folder_path):
                images_png_path = os.path.join(folder_path, "*.{}".format('png'))
                image_jpg_path = os.path.join(folder_path, "*.{}".format('jpg'))
                images_png = glob.glob(images_png_path)
                images_jpg = glob.glob(image_jpg_path)
                self.images.extend(images_png)
                self.images.extend(images_jpg)  

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_file = self.images[item]
        label_name = os.path.basename(os.path.dirname(image_file))
        image = default_loader(image_file)
        if self.data_transforms is not None:
            image = self.data_transforms(image)

        return image, int(label_name)