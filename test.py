import argparse
import time
import os
import sys
import logging
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from datasets import EllipseDataset
from models import SmallNet
from utils import inter_to_final_params


def main():
    # arguments
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument('--data_dir', type=str, default='images', help='path to data directory')
    parser.add_argument('--out_dir', type=str, default='output_test', help='path to output directory')
    parser.add_argument('--model_load_path', type=str, default='trained_best.pt', help='path used to load the model')

    # dataset
    parser.add_argument('--spatial_dim', type=int, default=50, help='spatial dimension of image (dim = width = height)')

    # training parameters
    parser.add_argument('--batch', type=int, default=64, help='batch size')

    # general
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()

    # output dir
    args.out_dir = '-'.join([args.out_dir, time.strftime("%Y%m%d-%H%M%S")])

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.out_dir, 'log_test.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", args)

    # reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    test_dataset = EllipseDataset(data_dir=args.data_dir, phase="test", dim=args.spatial_dim)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, pin_memory=True)

    model = SmallNet()
    model.load_state_dict(torch.load(args.model_load_path))

    model.eval()

    total = 0
    correct = 0

    # sum of absolute errors over test dataset
    sae_loss = torch.nn.L1Loss(reduction='sum')
    # absolute errors over test dataset, used to compute angle error
    ae_loss = torch.nn.L1Loss(reduction='none')
    final_sae = [0 for _ in range(5)]
    final_sae_angle = 0

    for step, sample in enumerate(test_loader):
        with torch.no_grad():
            output_class, output_params = model(sample["image"])

            # classification accuracy computations
            _, class_preds = torch.max(output_class, 1)
            total += class_preds.size(0)
            correct += (class_preds == sample["class"]).sum().item()

            # convert model outputted params to final params
            output_final_params = inter_to_final_params(output_params, args.spatial_dim)

            # look at ellipses parameters (not lines)
            ellipses_target_org_params = sample['final_params'][sample["class"] == 1]
            ellipses_output_org_params = output_final_params[sample["class"] == 1]

            # regression sae computations
            final_sae_batch = [sae_loss(ellipses_output_org_params[:, i], ellipses_target_org_params[:, i]).float() for
                               i in range(5)]
            final_sae = [final_sae[i] + final_sae_batch[i] for i in range(5)]

            # angle error computation, taking angles range into account
            final_ae_angle_batch = np.array(ae_loss(ellipses_output_org_params[:, 4], ellipses_target_org_params[:, 4]).float().numpy())
            final_ae_angle_batch = 90 - abs(final_ae_angle_batch - 90)
            final_sae_angle += sum(final_ae_angle_batch)

    # metrics
    acc = 100 * correct / total
    final_mae = [x.item() / len(test_loader.dataset) for x in final_sae]  # sum (sae) to average (mae)
    final_mae_angle = final_sae_angle / len(test_loader.dataset)
    test_metrics = {'acc': acc, 'mae': final_mae}

    logging.info(f"test acc {test_metrics['acc']:.2f}, mean absolute errors: center_x {test_metrics['mae'][0]:.2f}, "
                 f"center_y {test_metrics['mae'][1]:.2f}, axis_1 {test_metrics['mae'][2]:.2f}, "
                 f"axis_2 {test_metrics['mae'][3]:.2f}, angle {final_mae_angle:.2f}, "
                 f"angle error {final_mae_angle*100/180:.2f}%")


if __name__ == "__main__":
    main()
