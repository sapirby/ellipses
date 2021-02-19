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


def main():
    # arguments
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument('--data_dir', type=str, default='images', help='path to data directory')
    parser.add_argument('--out_dir', type=str, default='output_train', help='path to output directory')
    parser.add_argument('--model_save_path', type=str, default='.', help='path used to save the model')

    # dataset
    parser.add_argument('--spatial_dim', type=int, default=50, help='spatial dimension of image (dim = width = height)')
    parser.add_argument('--val', type=float, default=0.1, help='validation set percentage')

    # training parameters
    parser.add_argument('--epochs', type=int, default=40, help='number of training epochs')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--save_every', type=int, default=10, help='save model every x epochs')

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
    fh = logging.FileHandler(os.path.join(args.out_dir, 'log_train.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", args)

    # reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # datasets
    dataset = EllipseDataset(data_dir=args.data_dir, phase="train", dim=args.spatial_dim)
    val_len = int(args.val*len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               lengths=[len(dataset) - val_len, val_len],
                                                               generator=torch.Generator().manual_seed(args.seed))
    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=True, pin_memory=True)

    model = SmallNet()

    criterion_class = torch.nn.CrossEntropyLoss()

    criterion_reg = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)  # betas=(0.9, 0.999)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    writer = SummaryWriter(args.out_dir)

    logging.info("started training.")
    loss_min = sys.float_info.max
    for epoch in range(args.epochs):

        train_metrics = train(train_loader, model, criterion_class, criterion_reg, optimizer, epoch, writer)

        val_metrics = infer(val_loader, model, criterion_class, criterion_reg, epoch, writer)

        logging.info(f"epoch {epoch}: train acc {train_metrics['acc']:.2f}, train loss {train_metrics['loss']:.2f}, "
                     f"validation acc {val_metrics['acc']:.2f}, validation loss {val_metrics['loss']:.2f}")

        writer.add_scalar('learning_rate',
                          scheduler.get_last_lr()[0],
                          epoch)

        scheduler.step()

        if not (epoch + 1) % args.save_every:
            torch.save(model.state_dict(), os.path.join(args.model_save_path, 'trained_epoch_' + str(epoch+1) + '.pt'))

        if val_metrics['loss'] < loss_min:
            loss_min = val_metrics['loss']
            torch.save(model.state_dict(), os.path.join(args.model_save_path, 'trained_best.pt'))

            logging.info(f"saving new best model")

    writer.close()
    logging.info("finished training.")

    torch.save(model.state_dict(), os.path.join(args.model_save_path, 'trained_last.pt'))


def train(train_loader, model, criterion_class, criterion_reg, optimizer, epoch, writer):

    model.train()

    running_loss = 0.0
    total = 0
    correct = 0

    for step, sample in enumerate(train_loader):
        optimizer.zero_grad()

        output_class, output_params = model(sample["image"])
        loss_class = criterion_class(output_class, sample["class"])

        # calc loss for params only for ellipses
        ellipses_target_params = sample['inter_params'][sample["class"] == 1]
        ellipses_output_params = output_params[sample["class"] == 1]

        losses_reg = [criterion_reg(ellipses_output_params[:,i], ellipses_target_params[:,i]).float()
                      for i in range(output_params.shape[1])]
        losses_reg_weighted = [loss_reg * weight for loss_reg, weight in zip(losses_reg, [10, 10, 10, 10, 1, 1])]
        loss = loss_class + sum(losses_reg_weighted)
        loss.backward()
        running_loss += loss.item()

        optimizer.step()

        # compute accuracy
        _, class_preds = torch.max(output_class, 1)
        total += class_preds.size(0)
        correct += (class_preds == sample["class"]).sum().item()

        writer.add_scalar('training loss',
                          loss,
                          epoch * len(train_loader) + step)

        writer.add_scalar('training loss class',
                          loss_class,
                          epoch * len(train_loader) + step)

        for loss_reg_value, loss_reg_name in zip(losses_reg, ['center_x', 'center_y', 'axis_1', 'axis_2', 'cos', 'sin']):
            writer.add_scalar('training loss ' + loss_reg_name,
                              loss_reg_value,
                              epoch * len(train_loader) + step)

    # metrics
    acc = 100 * correct / total
    writer.add_scalar('training accuracy', acc, epoch * len(train_loader) + step)

    metrics = {'acc': acc, 'loss': running_loss / step}

    return metrics


def infer(val_loader, model, criterion_class, criterion_reg, epoch, writer):

    model.eval()

    running_loss = 0.0
    total = 0
    correct = 0

    for step, sample in enumerate(val_loader):
        with torch.no_grad():
            output_class, output_params = model(sample["image"])
            loss_class = criterion_class(output_class, sample["class"])

            # look at ellipses parameters (not lines)
            ellipses_target_params = sample['inter_params'][sample["class"] == 1]
            ellipses_output_params = output_params[sample["class"] == 1]
            losses_reg = [criterion_reg(ellipses_output_params[:, i], ellipses_target_params[:, i]).float() for i in
                          range(output_params.shape[1])]

            loss = loss_class + sum(losses_reg)

            running_loss += loss.item()

            # classification accuracy computations
            _, class_preds = torch.max(output_class, 1)
            total += class_preds.size(0)
            correct += (class_preds == sample["class"]).sum().item()

            writer.add_scalar('validation loss',
                              loss,
                              epoch * len(val_loader) + step)

            writer.add_scalar('validation loss class',
                              loss_class,
                              epoch * len(val_loader) + step)

            for loss_reg_value, loss_reg_name in zip(losses_reg, ['center_x', 'center_y', 'axis_1', 'axis_2', 'cos', 'sin']):
                writer.add_scalar('validation loss ' + loss_reg_name,
                                  loss_reg_value,
                                  epoch * len(val_loader) + step)

    # metrics
    acc = 100 * correct / total
    writer.add_scalar('validation accuracy', acc, epoch * len(val_loader) + step)

    metrics = {'acc': acc, 'loss': running_loss / step}

    return metrics


if __name__ == '__main__':
    main()



