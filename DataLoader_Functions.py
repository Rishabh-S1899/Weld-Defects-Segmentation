import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import imageio
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def show_img(Imagepath):
    # base_dir=r'Combined_RIAWELC_Dataset\training'
    # dir=os.path.join(base_dir,Imagepath)
    image=imageio.imread(Imagepath)
    print("The dimensions of the shape are: " , image.shape)
    plt.imshow(image, cmap="gray") 


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    def __call__(self, sample):
        image, Defects = sample['image'], sample['Defects']
        # h, w = image.shape[:2]
        # if isinstance(self.output_size, int):
        #     if h > w:
        #         new_h, new_w = self.output_size * h / w, self.output_size
        #     else:
        #         new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        #     new_h, new_w = self.output_size

        # new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (self.output_size, self.output_size))

        # h and w are swapped for Defects because for images,
        # x and y axes are axis 1 and 0 respectively
        # Defects = Defects * [new_w / w, new_h / h]

        return {'image': img, 'Defects': Defects}
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, Defects = sample['image'], sample['Defects']

        # Check if the image has a channel dimension
        # if len(image.shape) == 2:
        #     # Grayscale image, add a channel dimension
        #     image = image[:, :, None]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        # image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'Defects': torch.from_numpy(Defects)}


class WeldDefectXRayDataSet(Dataset):
    """Weld Defect X-Ray dataset."""

    def __init__(self, csv_file, img_root_dir, transform=None):
        
        self.labels=pd.read_csv(csv_file)
        self.img_dir=img_root_dir
        self.transform=transform
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) :
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name=os.path.join(self.img_dir,self.labels.iloc[idx]['Filename'])
        image=io.imread(img_name)
        Defects = self.labels.iloc[idx]['One-Hot-Encoding']
        Defects=np.array(Defects.strip('[]').split(','), dtype=float)
        # Defects = np.array([Defects], dtype=str).reshape(-1, 2)
        sample = {'image': image, 'Defects': Defects, 'img_name': img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample
        


# X_ray_dataset=WeldDefectXRayDataSet(csv_file='Processed_train copy.csv',img_root_dir=r'Combined_RIAWELC_Dataset\training')

# fig = plt.figure()

# for i, sample in enumerate(X_ray_dataset):
#     print(i, sample['image'].shape, sample['Defects'].shape)

#     # ax = plt.subplot(1, 4, i + 1)
#     # plt.tight_layout()
#     # ax.set_title('Sample #{}'.format(i))
#     # ax.axis('off')
#     # show_img(sample['img_name'])

#     if i == 4:
#         # plt.show()
#         break


# transformed_dataset = WeldDefectXRayDataSet(csv_file='Processed_train copy.csv',
#                                            img_root_dir=r'Combined_RIAWELC_Dataset\training',
#                                            transform=transforms.Compose([
#                                                Rescale(224),
#                                                ToTensor()
#                                            ]))

# for i, sample in enumerate(transformed_dataset):
#     print(i, sample['image'].size(), sample['Defects'])

#     if i == 3:
#         break

# dataloader = DataLoader(transformed_dataset, batch_size=1,
#                         shuffle=True, num_workers=0)


# # Helper function to show a batch
# def show_Defects_batch(sample_batched):
#     """Show image with Defects for a batch of samples."""
#     images_batch, Defects_batch = \
#             sample_batched['image'], sample_batched['Defects']
#     batch_size = len(images_batch)
#     im_size = images_batch.size(2)
#     grid_border_size = 2

#     # grid = utils.make_grid(images_batch)
#     # plt.imshow(grid.numpy().transpose((1, 2, 0)))

#     for i in range(batch_size):
#         # plt.scatter(Defects_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
#         #             Defects_batch[i, :, 1].numpy() + grid_border_size,
#         #             s=10, marker='.', c='r')
#         plt.imshow(images_batch[i],cmap='gray')
#         plt.title('Batch from dataloader')

# # if you are using Windows, uncomment the next line and indent the for loop.
# # you might need to go back and change ``num_workers`` to 0.

# # if __name__ == '__main__':
# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['Defects'].size())

#     # observe 4th batch and stop.
#     if i_batch == 1:
#         plt.figure()
#         show_Defects_batch(sample_batched)
#         plt.axis('off')
#         plt.ioff()
#         plt.show()
#         break