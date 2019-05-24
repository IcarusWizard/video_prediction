import torch
import torchvision
import numpy as np

from Video_dataset import VideoDataset
from model_util import setup_seed, mse_loss, save_model, load_model
from data_util import torch_save_gif
from models import CDNA

import argparse, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/bair')
    parser.add_argument('--model_path', type=str, default='model/bair')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu_workers', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='cdna')
    parser.add_argument('--start_point', type=int, default=0)

    args = parser.parse_args()

    setup_seed(args.seed)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    train_set = VideoDataset(args.data_path, 'train', args.horizon, fix_start=False)
    val_set = VideoDataset(args.data_path, 'val', args.horizon, fix_start=True)

    config = train_set.get_config()
    H, W, C = config['observations']
    A = config['actions'][0]
    T = args.horizon

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.cpu_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers=args.cpu_workers)

    if args.model_name == 'cdna':
        model = CDNA(T, H, W, C, A)

    if args.start_point > 0:
        load_model(os.path.join(args.model_path, '{}_{}.pt'.format(args.model_name, args.start_point)), eval_mode=False)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch = args.start_point
    while epoch < args.start_point + args.epoch:
        for j, data in enumerate(train_loader):
            observations = data['observations']
            actions = data['actions']

            # B x T ==> T x B
            observations = torch.transpose(observations, 0, 1)
            actions = torch.transpose(actions, 0, 1)

            predicted_observations = model(observations[0], actions)

            loss = mse_loss(observations, predicted_observations) / args.batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            opt.step()
            opt.zero_grad()

        epoch += 1
        save_model(model, os.path.join(args.model_path, '{}_{}.pt'.format(args.model_name, epoch)))

        gif_path = os.path.join(args.model_path, args.model_name, 'val_{}'.format(epoch))
        if not os.path.exists(gif_path):
            os.makedirs(gif_path)

        losses = []

        for j, data in enumerate(val_loader):
            observations = data['observations']
            actions = data['actions']

            # B x T ==> T x B
            observations = torch.transpose(observations, 0, 1)
            actions = torch.transpose(actions, 0, 1)

            predicted_observations = model(observations[0], actions)

            video = torch.cat([observations[0, 0].unsqueeze(0), predicted_observations[0 : T - 1, 0]])
            torch_save_gif(os.path.join(gif_path, "{}.gif".format(j)), video, fps=10)

            loss = mse_loss(observations, predicted_observations).item() / args.batch_size
            losses.append(loss)
            
            opt.zero_grad()

        print("-" * 50)
        print("In epoch {}, loss in val set is {}".format(epoch, np.mean(losses)))
        print("-" * 50)


if __name__ == '__main__':
    main()