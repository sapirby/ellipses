import torch
# import torchvision
# import matplotlib.pyplot as plt
# import numpy as np


def normalize_angle(angle, swapped=False):
    """
    Normalize angles from 0 - 359 degrees range to 0 - 179 degrees range.
    if axis were swapped, update angle by 90 degrees.
    :param angle: angle to normalize
    :param swapped: if axis were swapped
    :return:
    """

    if swapped:
        if angle < 270:
            angle += 90
        else:
            angle -= 90

    if 180 <= angle < 360:
        angle -= 180

    return angle


def inter_to_final_params(inter_params_tensor, dim):
    """
    Convert intermediate parameters to final parameters
    :param inter_params_tensor: tensor of intermediate parameters:
                                center_x_norm, center_y_norm, axis_1_norm, axis_2_norm, cos, sin
    :return: tensor of final parameters:
             center_x, center_y, axis_1, axis_2, angle_norm
    """

    final_params = torch.zeros((inter_params_tensor.shape[0], 5))

    # scale center and axis by image dimension back to pixels
    final_params[:, 0:4] = inter_params_tensor[:, 0:4] * dim

    # use atan to compute angle from cosine and sine. verify final range 0 - 179 degrees.
    final_params[:, 4] = torch.rad2deg(torch.atan2(inter_params_tensor[:, 5], inter_params_tensor[:, 4]))
    final_params[:, 4] = torch.where(final_params[:, 4] >= 0, final_params[:, 4], final_params[:, 4] + 360)
    final_params[:, 4] = torch.where(final_params[:, 4] < 180, final_params[:, 4], final_params[:, 4] - 180)

    return final_params


# def plot_images(images_tensor):
#     images = torchvision.utils.make_grid(images_tensor)
#     plt.imshow(np.transpose(images.numpy(), (1, 2, 0)))
#     plt.show()


# def calc_img_stats(dataloader):#
#     mean = 0
#     std = 0
#     for sample_batched in dataloader:
#         images = sample_batched['image']
#         images = images.view(images.shape[0], images.shape[1], -1)
#         mean += images.mean(2).sum(0)
#         std += images.std(2).sum(0)
#
#     mean /= len(dataloader.dataset)
#     std /= len(dataloader.dataset)
#
#     return mean, std
