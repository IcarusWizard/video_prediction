import torch
import torchvision
from torch.functional import F

# --------------------------------------------------------
#                        Modules
# --------------------------------------------------------
class CNN_LSTM(torch.nn.Module):
    """
        you may refer to :
            https://github.com/febert/visual_mpc/blob/master/python_visual_mpc/video_prediction/lstm_ops12.py
    """
    def __init__(self, H, W, C, output_channels, filter_size, forget_bias=1.0):
        super().__init__()
        self.H = H
        self.W = W
        self.C = C
        self.output_channels = output_channels
        self.filter_size = filter_size
        self.forget_bias = forget_bias

        # Initiailize your layers here

        raise NotImplementedError

    def initialize_h(self):
        # get the initial value for h

        raise NotImplementedError

    def forward(self, inputs, h=None):
        """
            Inputs:
                inputs -> tensor[B, C, H, W]
                h -> tensor[B, 2 * output_channels, H, W]
            Outputs:
                outputs -> tensor[B, output_chanels, H, W]
                new_h -> tensor[B, 2 * output_channels, H, W]
        """
        if h == None:
            h = self.initialize_h()

        # do forward computation here

        raise NotImplementedError

        return outputs, new_h

class Encoder(torch.nn.Module):
    def __init__(self, H=64, W=64, C=3, filter_size=5):
        super().__init__()
        self.H = H
        self.W = W
        self.C = C
        self.filter_size = filter_size
        self.padding = filter_size // 2

        self.conv1 = torch.nn.Conv2d(self.C, 32, kernel_size=self.filter_size, stride=2, padding=self.padding)
        self.norm1 = torch.nn.LayerNorm((32, self.H // 2, self.W // 2))
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.norm2 = torch.nn.LayerNorm((32, self.H // 2, self.W // 2))
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.norm3 = torch.nn.LayerNorm((32, self.H // 2, self.W // 2))
        self.conv4 = torch.nn.Conv2d(32, 64, kernel_size=self.filter_size, stride=2, padding=self.padding)
        self.norm4 = torch.nn.LayerNorm((64, self.H // 4, self.W // 4))
        self.conv5 = torch.nn.Conv2d(64, 64, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.norm5 = torch.nn.LayerNorm((64, self.H // 4, self.W // 4))
        self.conv6 = torch.nn.Conv2d(64, 128, kernel_size=self.filter_size, stride=2, padding=self.padding)

    def forward(self, observation):
        """
            Input:
                observation -> tensor[B, C, H, W]
            Output:
                state -> [B, 128, H // 8, W // 8]
        """
        net = self.norm1(F.relu(self.conv1(observation)))
        net = self.norm2(F.relu(self.conv2(net)))
        net = self.norm3(F.relu(self.conv3(net)))
        net = self.norm4(F.relu(self.conv4(net)))
        net = self.norm5(F.relu(self.conv5(net)))
        state = self.conv6(net)

        return state

class StateTransform(torch.nn.Module):
    def __init__(self, H=8, W=8, C=128, A=4, filter_size=5):
        super().__init__()
        self.H = H
        self.W = W
        self.C = C
        self.A = A
        self.filter_size = filter_size
        self.padding = self.filter_size // 2

        self.conv1 = torch.nn.Conv2d(C + A, 2 * C, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.conv2 = torch.nn.Conv2d(2 * C, C, kernel_size=3, stride=1, padding=1)

    def forward(self, state, action):
        """
            Input:
                state -> tensor[B, C, H, W]
                action -> tensor[B, A]
            Output:
                next_state -> tensor[B, C, H, W]
        """
        # build the action in image form
        action_tile = action.view(action.shape[0], self.A, 1, 1)
        action_tile = action_tile.repeat(1, 1, self.H, self.W)

        state_action = torch.cat([state, action_tile], 1)

        return self.conv2(F.relu(self.conv1(state_action)))

class Decoder(torch.nn.Module):
    def __init__(self, H=8, W=8, C=3, filter_size=5):
        super().__init__()
        self.H = H
        self.W = W
        self.C = C
        self.filter_size = filter_size
        self.padding = self.filter_size // 2

        self.deconv1 = torch.nn.ConvTranspose2d(128, 64, 
                                kernel_size=self.filter_size, stride=2, padding=self.padding, output_padding=1)
        self.conv1 = torch.nn.Conv2d(64, 64, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.deconv2 = torch.nn.ConvTranspose2d(64, 32, 
                                kernel_size=self.filter_size, stride=2, padding=self.padding, output_padding=1) 
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=self.filter_size, stride=1, padding=self.padding)  
        self.deconv3 = torch.nn.ConvTranspose2d(32, 16, 
                                kernel_size=self.filter_size, stride=2, padding=self.padding, output_padding=1)       
        self.conv3 = torch.nn.Conv2d(16, 16, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.conv_top = torch.nn.Conv2d(16, self.C, kernel_size=self.filter_size, stride=1, padding=self.padding)

    def forward(self, state):
        """
            Input:
                state -> tensor[B, 128, H, W]
            Output:
                observation -> tensor[B, C, H * 8, W * 8]
        """
        upsample1 = F.relu(self.deconv1(state))
        upsample1 = F.relu(self.conv1(upsample1))
        upsample2 = F.relu(self.deconv2(upsample1))
        upsample2 = F.relu(self.conv2(upsample2))
        upsample3 = F.relu(self.deconv3(upsample2))
        upsample3 = F.relu(self.conv3(upsample3))

        observation = self.conv_top(upsample3)

        return observation

class EncoderSkip(torch.nn.Module):
    def __init__(self, H=64, W=64, C=3, filter_size=5):
        super().__init__()
        self.H = H
        self.W = W
        self.C = C
        self.filter_size = filter_size
        self.padding = filter_size // 2

        self.conv1 = torch.nn.Conv2d(self.C, 32, kernel_size=self.filter_size, stride=2, padding=self.padding)
        self.norm1 = torch.nn.LayerNorm((32, self.H // 2, self.W // 2))
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.norm2 = torch.nn.LayerNorm((32, self.H // 2, self.W // 2))
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.norm3 = torch.nn.LayerNorm((32, self.H // 2, self.W // 2))
        self.conv4 = torch.nn.Conv2d(32, 64, kernel_size=self.filter_size, stride=2, padding=self.padding)
        self.norm4 = torch.nn.LayerNorm((64, self.H // 4, self.W // 4))
        self.conv5 = torch.nn.Conv2d(64, 64, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.norm5 = torch.nn.LayerNorm((64, self.H // 4, self.W // 4))
        self.conv6 = torch.nn.Conv2d(64, 128, kernel_size=self.filter_size, stride=2, padding=self.padding)

    def forward(self, observation):
        """
            Input:
                observation -> tensor[B, C, H, W]
            Output:
                state -> tensor[B, 128, H // 8, W // 8]
                en1 -> tensor[B, 32, H // 2, W // 2]
                en2 -> tensor[B, 64, H // 4, W // 4]
        """
        en1 = self.norm1(F.relu(self.conv1(observation)))
        en1 = self.norm2(F.relu(self.conv2(en1)))
        en1 = self.norm3(F.relu(self.conv3(en1)))
        en2 = self.norm4(F.relu(self.conv4(en1)))
        en2 = self.norm5(F.relu(self.conv5(en2)))
        state = self.conv6(en2)

        return state, en1, en2

class _DecoderSkip(torch.nn.Module):
    def __init__(self, H=8, W=8, C=3, filter_size=5):
        super().__init__()
        self.H = H
        self.W = W
        self.C = C
        self.filter_size = filter_size
        self.padding = self.filter_size // 2

        self.deconv1 = torch.nn.ConvTranspose2d(128, 64, 
                                kernel_size=self.filter_size, stride=2, padding=self.padding, output_padding=1)
        self.conv1_1 = torch.nn.Conv2d(64, 64, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.conv1_skip = torch.nn.Conv2d(64 * 2, 64, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.conv1_norm = torch.nn.LayerNorm((64, H * 2, W * 2))
        self.conv1_skip_norm = torch.nn.LayerNorm((64, H * 2, W * 2))

        self.deconv2 = torch.nn.ConvTranspose2d(64, 32, 
                                kernel_size=self.filter_size, stride=2, padding=self.padding, output_padding=1) 
        self.conv2_1 = torch.nn.Conv2d(32, 32, kernel_size=self.filter_size, stride=1, padding=self.padding)  
        self.conv2_2 = torch.nn.Conv2d(32, 32, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.conv2_skip = torch.nn.Conv2d(32 * 2, 32, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.conv2_norm = torch.nn.LayerNorm((32, H * 4, W * 4))
        self.conv2_skip_norm = torch.nn.LayerNorm((32, H * 4, W * 4))

        self.deconv3 = torch.nn.ConvTranspose2d(32, 16, 
                                kernel_size=self.filter_size, stride=2, padding=self.padding, output_padding=1)       
        self.conv3 = torch.nn.Conv2d(16, 16, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.conv_top = torch.nn.Conv2d(16, self.C, kernel_size=self.filter_size, stride=1, padding=self.padding)

    def forward(self, state, en1, en2):
        """
            Input:
                state -> tensor[B, 128, H, W]
                en1 -> tensor[B, 64, H * 2, W * 2]
                en2 -> tensor[B, 32, H * 4, W * 4]
            Output:
                observation -> tensor[B, C, H * 8, W * 8]
                en1 -> tensor[B, 64, H * 2, W * 2]
                en2 -> tensor[B, 32, H * 4, W * 4]
        """
        upsample1 = F.relu(self.deconv1(state))

        cat1 = torch.cat([upsample1, en2], 1)
        en2 = self.conv1_skip_norm(F.relu(self.conv1_skip(cat1)))
        upsample1 = (self.conv1_norm(F.relu(self.conv1_1(upsample1))) + en2) / 2.0
        upsample1 = F.relu(self.conv1_2(upsample1))
        # upsample1 = F.relu(self.conv1(upsample1)) + en2      
        upsample2 = F.relu(self.deconv2(upsample1))

        cat2 = torch.cat([upsample2, en1], 1)
        en1 = self.conv2_skip_norm(F.relu(self.conv2_skip(cat2)))
        upsample2 = (self.conv2_norm(F.relu(self.conv2_1(upsample2))) + en1) / 2.0
        upsample2 = F.relu(self.conv2_2(upsample2))
        # upsample2 = F.relu(self.conv2(upsample2)) + en1
        upsample3 = F.relu(self.deconv3(upsample2))

        upsample3 = F.relu(self.conv3(upsample3))

        observation = self.conv_top(upsample3)

        return observation, en1, en2

class DecoderSkip(torch.nn.Module):
    def __init__(self, H=8, W=8, C=3, filter_size=5):
        super().__init__()
        self.H = H
        self.W = W
        self.C = C
        self.filter_size = filter_size
        self.padding = self.filter_size // 2

        self.deconv1 = torch.nn.ConvTranspose2d(128, 64, 
                                kernel_size=self.filter_size, stride=2, padding=self.padding, output_padding=1)
        self.conv1_1 = torch.nn.Conv2d(64 * 2, 64, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.conv1_skip = torch.nn.Conv2d(64 * 2, 64, kernel_size=self.filter_size, stride=1, padding=self.padding)

        self.deconv2 = torch.nn.ConvTranspose2d(64, 32, 
                                kernel_size=self.filter_size, stride=2, padding=self.padding, output_padding=1) 
        self.conv2_1 = torch.nn.Conv2d(32, 32, kernel_size=self.filter_size, stride=1, padding=self.padding)  
        self.conv2_2 = torch.nn.Conv2d(32, 32, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.conv2_skip = torch.nn.Conv2d(32 * 2, 32, kernel_size=self.filter_size, stride=1, padding=self.padding)

        self.deconv3 = torch.nn.ConvTranspose2d(32, 16, 
                                kernel_size=self.filter_size, stride=2, padding=self.padding, output_padding=1)       
        self.conv3 = torch.nn.Conv2d(16, 16, kernel_size=self.filter_size, stride=1, padding=self.padding)
        self.conv_top = torch.nn.Conv2d(16, self.C, kernel_size=self.filter_size, stride=1, padding=self.padding)

    def forward(self, state, en1, en2):
        """
            Input:
                state -> tensor[B, 128, H, W]
                en1 -> tensor[B, 64, H * 2, W * 2]
                en2 -> tensor[B, 32, H * 4, W * 4]
            Output:
                observation -> tensor[B, C, H * 8, W * 8]
                en1 -> tensor[B, 64, H * 2, W * 2]
                en2 -> tensor[B, 32, H * 4, W * 4]
        """
        upsample1 = F.relu(self.deconv1(state))

        cat1 = torch.cat([upsample1, en2], 1)
        en2 = self.conv1_skip(cat1)
        upsample1 = F.relu(self.conv1_1(cat1))
        upsample1 = F.relu(self.conv1_2(upsample1))    
        upsample2 = F.relu(self.deconv2(upsample1))

        cat2 = torch.cat([upsample2, en1], 1)
        en1 = self.conv2_skip(cat2)
        upsample2 = F.relu(self.conv2_1(cat2))
        upsample2 = F.relu(self.conv2_2(upsample2))
        upsample3 = F.relu(self.deconv3(upsample2))

        upsample3 = F.relu(self.conv3(upsample3))

        observation = self.conv_top(upsample3)

        return observation, en1, en2

# --------------------------------------------------------
#                        Models
# --------------------------------------------------------
class VideoPrediction(torch.nn.Module):
    def __init__(self, T, H, W, C, A):
        """
            Inputs:
                T -> int : sequence length
                H -> int : height of the image
                W -> int : width of the image
                C -> int : channel of the image
                A -> int : action space size
        """
        super().__init__()
        self.T = T
        self.H = H
        self.W = W
        self.C = C
        self.A = A

    def forward(self, observation_0, actions):
        """
            Inputs:
                observations_0 -> tensor[B, C, H, W]: the first observation in the sequence (only real image)
                actions -> tensor[T, B, A] : action sequence the robot take, i.e. a_0 ..... a_(T-1)
            Outputs:
                predicted_observations -> tensor[T, B, C, H, W] :
                    the predicted observation sequence given the actions, i.e. o_1 ..... o_T
        """
        raise NotImplementedError

class CDNA(VideoPrediction):
    """
        you may refer to :
            https://github.com/alexlee-gk/video_prediction/blob/master/video_prediction/models/dna_model.py
    """
    def __init__(self, T, H, W, C, A):
        super(CDNA, self).__init__(T, H, W, C, A)
        
        # Initialize your network here

        raise NotImplementedError

    def forward(self, observation_0, actions):
        last_observation = observation_0

        predicted_observations = []

        for t in self.T:
            action = actions[t]

            # Implement the prediction here
            
            prediction = None

            raise NotImplementedError

            predicted_observations.append(prediction.unsqueeze(0))
            last_observation = prediction

        return torch.cat(predicted_observations, 0)

class ETD(torch.nn.Module):
    def __init__(self, H, W, C, A, T, filter_size):
        super().__init__()
        self.H = H
        self.W = W
        self.C = C
        self.A = A
        self.T = T
        self.filter_size = filter_size

        self.encoder = Encoder(H, W, C, filter_size)
        self.transform = StateTransform(H // 8, W // 8, 128, A, filter_size)
        self.decoder = Decoder(H // 8, W // 8, C, filter_size)

    def forward(self, observation_0, actions):
        predicted_observations = []

        last_state = self.encoder(observation_0)

        for t in range(self.T):
            action = actions[t]
            new_state = self.transform(last_state, action)
            prediction = self.decoder(new_state)
            last_state = new_state
            predicted_observations.append(prediction.unsqueeze(0))

        return torch.cat(predicted_observations, 0)

class ETDS(torch.nn.Module):
    def __init__(self, H, W, C, A, T, filter_size):
        super().__init__()
        self.H = H
        self.W = W
        self.C = C
        self.A = A
        self.T = T
        self.filter_size = filter_size

        self.encoder = EncoderSkip(H, W, C, filter_size)
        self.transform = StateTransform(H // 8, W // 8, 128, A, filter_size)
        self.decoder = DecoderSkip(H // 8, W // 8, C, filter_size)

    def forward(self, observation_0, actions):
        predicted_observations = []

        last_state, en1, en2 = self.encoder(observation_0)

        for t in range(self.T):
            action = actions[t]
            new_state = self.transform(last_state, action)
            prediction, en1, en2 = self.decoder(new_state, en1, en2)
            last_state = new_state
            predicted_observations.append(prediction.unsqueeze(0))

        return torch.cat(predicted_observations, 0)