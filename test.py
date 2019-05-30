import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from Video_dataset import VideoDataset
from model_util import setup_seed, mse_loss, save_model, load_model
from data_util import torch_save_gif
from models import CDNA, ETD, ETDS, ETDM, ETDSD

import argparse, os, time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/bair')
    parser.add_argument('--model_path', type=str, default='model/bair')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--cpu_workers', type=int, default=4)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='cdna')
    parser.add_argument('--load_point', type=int, default=10)
    parser.add_argument('--no-gif', dest='save_gif', action='store_false')
    parser.set_defaults(save_gif=True)

    args = parser.parse_args()

    device = 'cuda:%d' % args.gpu_id if torch.cuda.device_count() > 0 else 'cpu'

    # dataset setup
    test_set = VideoDataset(args.data_path, 'test', args.horizon, fix_start=True)

    config = test_set.get_config()
    H, W, C = config['observations']
    A = config['actions'][0]
    T = args.horizon

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, num_workers=args.cpu_workers)

    # model setup
    if args.model_name == 'cdna':
        model = CDNA(T, H, W, C, A)
    elif args.model_name == 'etd':
        model = ETD(H, W, C, A, T, 5)
    elif args.model_name == 'etds':
        model = ETDS(H, W, C, A, T, 5)
    elif args.model_name == 'etdm':
        model = ETDM(H, W, C, A, T, 5)
    elif args.model_name == 'etdsd':
        model = ETDSD(H, W, C, A, T, 5)

    model.to(device)

    model_path = os.path.join(args.model_path, '{}_10'.format(args.model_name))

    load_model(model, os.path.join(model_path, '{}_{}.pt'.format(args.model_name, args.load_point)), eval_mode=True)

    # tensorboard
    writer = SummaryWriter()


    gif_path = os.path.join(model_path, 'test_{}'.format(args.horizon))
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)

    losses = []
    videos = []
    inference_times = []

    for j, data in enumerate(test_loader):
        observations = data['observations']
        actions = data['actions']

        # B x T ==> T x B
        observations = torch.transpose(observations, 0, 1).to(device)
        actions = torch.transpose(actions, 0, 1).to(device)

        start_time = time.time()
        predicted_observations = model(observations[0], actions)
        inference_times.append((time.time() - start_time) * 1000)

        video = torch.cat([observations[0, 0].unsqueeze(0), predicted_observations[0 : T - 1, 0]]) # tensor[T, C, H, W]
        videos.append(video.unsqueeze(0).detach().cpu())

        if args.save_gif:
            torch_save_gif(os.path.join(gif_path, "{}.gif".format(j)), video.detach().cpu(), fps=10)

        loss = mse_loss(observations, predicted_observations).item() / args.batch_size
        losses.append(loss)

        del loss, observations, actions, predicted_observations # clear the memory

    
    videos = torch.cat(videos, 0)
    writer.add_video('test_video_{}_{}'.format(args.model_name, args.horizon), videos, global_step=0, fps=10)

    print("-" * 50)
    print("mean loss in test set is {}, std is {}".format(np.mean(losses), np.std(losses)))
    print("mean inference time in test set is {}, std is {}".format(np.mean(inference_times), np.std(inference_times)))
    print("-" * 50)


if __name__ == '__main__':
    main()
