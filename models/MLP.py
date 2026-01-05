from torch import nn

class MLP(nn.Module):
    def __init__(self,
                 num_layers,
                 neurons,
                 input_size):

        super().__init__()

        ls = [nn.Sequential(
            nn.Linear(input_size, neurons),
            nn.Tanh()
        )]

        for _ in range(num_layers):
            ls.append(nn.Sequential(
                nn.Linear(neurons, neurons),
                nn.Tanh()
            ))

        ls.append(nn.Sequential)