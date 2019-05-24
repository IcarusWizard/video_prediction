import torch
import torchvision

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
