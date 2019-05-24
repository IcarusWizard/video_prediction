import torch
import torchvision
import numpy as np

import pickle, os, random, time

from data_util import torch_save_gif

class VideoDataset(torch.utils.data.Dataset):
    """
        Return dataset contain video images
        Keys and values:
            observations : tensor B x T x C x H x W
            actions : tensor B x T x A
    """
    def __init__(self, 
                path,
                dataset,
                horizon,
                fix_start=False):
        super().__init__()
        self.horizon = horizon
        self.path = path
        self.dataset = dataset
        self.fix_start = fix_start
        self.config = self.load_pkl(os.path.join(path, 'config.pkl'))
        
        # use only one view
        self.keys = []
        self.shapes = []
        for key, shape in self.config.items():
            if 'image' in key:
                if 'image_main' in key or 'image_view0' in key:
                    self.image_key = key
                    self.image_shape = shape
                    self.keys.append(key)
                    self.shapes.append(shape)
            elif 'action' in key:
                self.action_key = key
                self.action_shape = shape
                self.keys.append(key)
                self.shapes.append(shape)
        self.config = {'observations' : self.image_shape, 
                        'actions' : self.action_shape}

        # load filelist
        filenames = sorted(os.listdir(os.path.join(path, dataset)))
        self.filelist = [os.path.join(path, dataset, filename) for filename in filenames]

        # find sequence length
        data = self.load_pkl(self.filelist[0])
        video = data[self.image_key]
        self.sequence_length = video.shape[0]

        assert self.horizon <= self.sequence_length, "horizon must smaller than sequence length, i.e. {} <= {}".format(
            self.horizon,
            self.sequence_length
        )
        
    def load_pkl(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def set_config(self, config):
        self.config = config

    def get_config(self):
        return self.config
        
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        # load data
        data = self.load_pkl(self.filelist[index])

        # set start point
        start = 0 if self.fix_start else random.randint(0, self.sequence_length - self.horizon)

        output = {}
        for key in self.keys:
            key_data = data[key][start : start + self.horizon]
            if key == self.image_key:
                key_data = key_data / 255
                key_data = np.transpose(key_data, (0, 3, 1, 2))
                output['observations'] = torch.from_numpy(key_data.astype(np.float32))
            elif key == self.action_key:
                output['actions'] = torch.from_numpy(key_data.astype(np.float32))
        
        return output

if __name__ == '__main__':
    dataset = VideoDataset('data/bair', 'val', horizon=15, fix_start=True)
    config = dataset.get_config()
    print(config)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=8, shuffle=False)
    count = 0
    start = time.time()
    for data in loader:
        end = time.time()
        print(end - start)
        start = end
        imgs = data['observations'][0]
        torch_save_gif('%03d_video.gif' % count, imgs, fps=10)
        count += 1
