import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

from utils import normalize_angle


class EllipseDataset(Dataset):
    def __init__(self, data_dir, phase, dim):
        if phase not in ['train', 'test']:
            raise NotImplementedError("Can't create EllipseDataset. Please choose phase 'train' or 'test'")
        self.data_dir = data_dir
        self.phase = phase
        self.dim = dim
        self.metadata = None
        self.image_transform = None
        self.update_metadata()
        self.update_transforms()

    def update_transforms(self, transform_mode='rgb'):
        """
        Update self.image_transform with transforms to operate on dataset images.
        :param transform_mode: mode of transform.
            'rgb' : transform to tensor.
            'rgb_norm' : transform to tensor and normalize.
            'grayscale_norm' : transform grayscale, transform to tensor and normalize.
        """

        if transform_mode == 'rgb':
            self.image_transform = transforms.Compose([transforms.ToTensor()])

        elif transform_mode == 'rgb_norm':
            self.image_transform = transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize((0.6043, 0.6047, 0.6042),
                                                                            (0.0443, 0.0425, 0.0448))])

        elif transform_mode == 'grayscale_norm':
            self.image_transform = transforms.Compose([transforms.Grayscale(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(0.6045, 0.0330)])
        else:
            raise NotImplementedError("Invalid transform_mode.")

    def update_metadata(self):
        """
        Read metadata from txt data file to data frame.
        Preprocess data:
            - Represent is_ellipse as class.
            - Set axis1, axis2 as major axis, minor axis accordingly.
            - Normalize angle to range 0 - 179 and compute sine and cosine of angle.
            - Normalize center and axes by image dimension (to range 0 - 1).
        Update self.metadata.
        """
        # read data
        metadata_file = os.path.join(self.data_dir, self.phase + '_data.txt')
        param_names = ['image', 'is_ellipse', 'center_x', 'center_y', 'angle', 'axis_1', 'axis_2']
        metadata = pd.read_csv(metadata_file, skiprows=1, sep=' |, ', engine='python', names=param_names)

        # preprocess data
        # represent is_ellipse as a class (1 for True, 0 for False)
        metadata['is_ellipse'] = metadata['is_ellipse'].astype(int)

        # swap axis_1, axis_2 if necessary, so that axis_1 = major axis (axis_2 = minor axis).
        # normalize angle accordingly to range 0 - 179 degrees
        metadata['axis_1'], metadata['axis_2'], metadata['angle_norm'] = \
            np.where(metadata['axis_2'] > metadata['axis_1'],
                     (metadata['axis_2'], metadata['axis_1'], metadata['angle'].apply(normalize_angle, swapped=True)),
                     (metadata['axis_1'], metadata['axis_2'], metadata['angle'].apply(normalize_angle)))

        # rearrange for convenience
        param_names.append('angle_norm')
        param_names.append(param_names.pop(4))
        metadata = metadata[param_names]

        # normalize center and axis by image dimension (output range: 0 - 1)
        metadata['center_x_norm'] = metadata['center_x'] / self.dim
        metadata['center_y_norm'] = metadata['center_y'] / self.dim

        metadata['axis_1_norm'] = metadata['axis_1'] / self.dim
        metadata['axis_2_norm'] = metadata['axis_2'] / self.dim

        # compute cos and sin in radians
        metadata['cos'] = np.cos(np.radians(metadata['angle_norm']))
        metadata['sin'] = np.sin(np.radians(metadata['angle_norm']))

        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Get sample from dataset. Each consists of:
         - image
         - class (1 - ellipse, 0 - not ellipse)
         - inter params: center_x_norm, center_y_norm, axis_1_norm, axis_2_norm, cos, sin
                         intermediate parameters, result of the preprocessing stage, more easily learned by model.
         - final params: center_x, center_y, axis_1, axis_2, angle_norm
                         final parameters, the parameters we want our final output to match.

        :param idx: sample inde
        """
        # image
        image_path = self.metadata.iloc[idx, 0]
        image = Image.open(image_path)
        image = self.image_transform(image)

        # class
        is_ellipse = np.array([self.metadata.iloc[idx, 1]]).squeeze()

        # final parameters: center_x, center_y, axis_1, axis_2, angle_norm
        final_params = np.array([self.metadata.iloc[idx, 2:7]], dtype=np.float32).squeeze()

        # intermediate parameters: center_x_norm, center_y_norm, axis_1_norm, axis_2_norm, cos, sin
        inter_params = np.array([self.metadata.iloc[idx, 8:]], dtype=np.float32).squeeze()

        sample = {'image': image,
                  'class': is_ellipse,
                  'inter_params': inter_params,
                  'final_params': final_params}

        return sample

    def print_data_stats(self):
        """
        Print metadata statistics
        :return:
        """
        pd.set_option('display.max_columns', None)
        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        print(f"head: \n{self.metadata.head()} \n")
        print(f"info: \n{self.metadata.info()} \n")
        print(f"Numerics stats: \n{self.metadata.describe()} \n")
        print(f"is_ellipse statistics: \n{self.metadata.is_ellipse.value_counts()} \n")


if __name__ == "__main__":
    dataset = EllipseDataset(data_dir="images", phase="train", dim=50)
    dataset.print_data_stats()
    dataset_iter = iter(dataset)
    sample = next(dataset_iter)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)



