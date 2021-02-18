# import torch
# from PIL import Image
# import numpy as np
# import torchvision.transforms.functional as F
#
#
# def horizontal_flip_and_edit_params(img, inter_params, final_params, dim=50, p=0.5):
#     if torch.rand(1) < p:
#
#         center_x, center_y, axis_1, axis_2, angle_norm = final_params
#
#         # intermediate parameters:
#         center_x_norm, center_y_norm, axis_1_norm, axis_2_norm, cos, sin = inter_params
#
#         img = F.hflip(img)
#
#         center_x = dim - center_x
#         center_x_norm = 1 - center_x_norm
#
#         if angle_norm < 90:
#             angle_norm += 90
#             temp = cos
#             cos = -sin
#             sin = temp
#         else:
#             angle_norm -= 90
#             temp = cos
#             cos = sin
#             sin = -temp
#
#         final_params = np.array([center_x, center_y, axis_1, axis_2, angle_norm], dtype=np.float32)
#
#         # intermediate parameters:
#         inter_params = np.array([center_x_norm, center_y_norm, axis_1_norm, axis_2_norm, cos, sin], dtype=np.float32)
#
#     return img, inter_params, final_params
