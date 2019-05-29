import torch
import torchvision
from torch.functional import F
from torch import nn

# --------------------------------------------------------
#                        Modules
# --------------------------------------------------------
class CNN_LSTM(nn.Module):
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
        self.conv = nn.Conv2d(
            self.C + self.output_channels,
            4 * self.output_channels, self.filter_size, padding=int(self.filter_size/2))

    def initialize_state(self, inputs):
        # get the initial value for hidden state
        batch_size = list(inputs.size())[0]
        state = torch.zeros(
            batch_size, 2 * self.output_channels, self.H, self.W, 
            device=inputs.device)
        return state

    def forward(self, inputs, state=None):
        """
            Inputs:
                inputs -> tensor[B, C, H, W]
                state -> tensor[B, 2 * output_channels, H, W]
            Outputs:
                outputs -> tensor[B, output_chanels, H, W]
                new_state -> tensor[B, 2 * output_channels, H, W]
        """
        
        

        if state is None:
            state = self.initialize_state(inputs)

        c, h = torch.split(state, self.output_channels, dim=1)
        # c = c.to(inputs.device)
        # h = h.to(inputs.device)
        inputs_h = torch.cat((inputs, h), 1)

        # do forward computation here
        i_j_f_o = F.relu(self.conv(inputs_h))

        i, j, f, outputs = torch.split(i_j_f_o, self.output_channels, dim=1)
        new_c = c * torch.sigmoid(f + self.forget_bias) + torch.sigmoid(i) + torch.tanh(j)
        new_h = torch.tanh(new_c) + torch.sigmoid(outputs)
        new_state = torch.cat((new_c, new_h), 1)

        return outputs, new_state

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
        self.conv2_1 = torch.nn.Conv2d(32 * 2, 32, kernel_size=self.filter_size, stride=1, padding=self.padding)  
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
        lstm_size = [32, 32, 64, 64, 128, 64, 32]
        self.enc0 = nn.Conv2d(3, 32, 5, stride=2, padding=2)
        self.lstm1 = CNN_LSTM(32, 32, 32, lstm_size[0], 5)
        self.lstm2 = CNN_LSTM(32, 32, lstm_size[0], lstm_size[1], 5)

        self.enc1 = nn.Conv2d(lstm_size[1], lstm_size[2], 3, stride=2, padding=1) 
        # Why is the kernel 3?
        self.lstm3 = CNN_LSTM(16, 16, lstm_size[2], lstm_size[2], 5)
        self.lstm4 = CNN_LSTM(16, 16, lstm_size[2], lstm_size[3], 5)

        self.enc2 = nn.Conv2d(lstm_size[3], lstm_size[3], 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(lstm_size[3] + 8, lstm_size[4], 1)

        self.lstm5 = CNN_LSTM(8, 8, lstm_size[4], lstm_size[4], 5) 
        self.enc4 = nn.ConvTranspose2d(lstm_size[4], lstm_size[5], 3, stride=2, padding=1, output_padding=1)
        self.lstm6 = CNN_LSTM(16, 16, lstm_size[5], lstm_size[5], 5)
        self.enc5 = nn.ConvTranspose2d(lstm_size[5] + lstm_size[2], lstm_size[6], 3, stride=2, padding=1, output_padding=1)
        self.lstm7 = CNN_LSTM(32, 32, lstm_size[6], lstm_size[6], 5)
        self.enc6 = nn.ConvTranspose2d(lstm_size[6] +lstm_size[0], 11, 3, stride=2, padding=1, output_padding=1)  
        self.enc7 = nn.Conv2d(11, 11, 1) 

        self.fc1 = nn.Linear(128 * 8 * 8, 10 * 5 * 5) 
        

    def forward(self, observation_0, actions):
        last_observation = observation_0
        state1, state2, state3, state4, state5, state6, state7 = None, None, None, None, None, None, None
        predicted_observations = []


        for t in range(self.T):
            action = actions[t] # action = [B, A]
            batch_size = action.shape[0]
            # Implement the prediction here
            enc0 = F.relu(self.enc0(last_observation))
            observation_1, state1 = self.lstm1(enc0, state1)
            observation_2, state2 = self.lstm2(observation_1, state2)
            enc1 = F.relu(self.enc1(observation_2))
            observation_3, state3 = self.lstm3(enc1, state3)
            observation_4, state4 = self.lstm4(observation_3, state4)
            
            enc2 = F.relu(self.enc2(observation_4))

            # Pass in action
            smear = action.view(batch_size, self.A, 1, 1)
            smear = smear.repeat(1, 2, 8, 8) # Tile action(Bx4x1x1) to Bx8x8x8
            enc2 = torch.cat((enc2, smear), 1)
            enc3 = F.relu(self.enc3(enc2))

            observation_5, state5 = self.lstm5(enc3, state5)
            enc4 = F.relu(self.enc4(observation_5))
            observation_6, state6 = self.lstm6(enc4, state6)

            # Skip connection
            observation_6 = torch.cat((observation_6, enc1), 1) 
            enc5 = F.relu(self.enc5(observation_6))
            observation_7, state7 = self.lstm7(enc5, state7)
            # Skip connection
            observation_7 = torch.cat((observation_7, enc0), 1) 
            enc6 = F.relu(self.enc6(observation_7))
            enc7 = F.relu(self.enc7(enc6))
            masks = F.softmax(enc7, dim=1)

            # CDNA kernels
            line = observation_5.view(-1, 128 * 8 * 8)
            linear_kernal = F.relu(self.fc1(line))
            kernels = linear_kernal.view(batch_size, 10, 5, 5)

            norm_factor = torch.sum(kernels, 1).view(-1, 1, 5, 5)
            kernels /= norm_factor
            images = []
            for b in range(batch_size):
                images.append(
                    F.conv2d(observation_0[b].unsqueeze(1), kernels[b].unsqueeze(1), padding=2
                    ).unsqueeze(0)) # ?
            transformed_images = torch.cat(images, 0)

            transformed_images.transpose(1, 2) # Change the dimention of channel and color channel
            
            transformed_images = torch.cat((transformed_images, observation_0.unsqueeze(2)), 2)
            image_r = transformed_images[:, 0, :, :, :].view(batch_size, 11, 64, 64)
            image_g = transformed_images[:, 1, :, :, :].view(batch_size, 11, 64, 64)
            image_b = transformed_images[:, 2, :, :, :].view(batch_size, 11, 64, 64)
            
            predicted_r = image_r * masks
            predicted_g = image_g * masks
            predicted_b = image_b * masks

            prediction = torch.cat([predicted_r.unsqueeze(1), predicted_g.unsqueeze(1), predicted_b.unsqueeze(1)], 1)
            prediction = sum(i for i in torch.unbind(prediction, dim=2))
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

class ETDM(torch.nn.Module):
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
        self.decoder_skip = DecoderSkip(H // 8, W // 8, C, filter_size)
        self.decoder_normal = Decoder(H // 8, W // 8, C, filter_size)
        self.decoder_mask = Decoder(H // 8, W // 8, 1, filter_size)

    def forward(self, observation_0, actions):
        predicted_observations = []

        last_state, en1, en2 = self.encoder(observation_0)

        for t in range(self.T):
            action = actions[t]
            new_state = self.transform(last_state, action)
            
            prediction_skip, en1, en2 = self.decoder_skip(new_state, en1, en2)
            prediction_normal = self.decoder_normal(new_state)
            prediction_mask = torch.sigmoid(self.decoder_mask(new_state))
            prediction = prediction_mask * prediction_normal + (1 - prediction_mask) * prediction_skip

            last_state = new_state
            predicted_observations.append(prediction.unsqueeze(0))

        return torch.cat(predicted_observations, 0)    

class ETDSD(torch.nn.Module):
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
        en1 = en1.detach()
        en2 = en2.detach()

        for t in range(self.T):
            action = actions[t]
            new_state = self.transform(last_state, action)
            prediction, en1, en2 = self.decoder(new_state, en1, en2)
            last_state = new_state
            predicted_observations.append(prediction.unsqueeze(0))

        return torch.cat(predicted_observations, 0)