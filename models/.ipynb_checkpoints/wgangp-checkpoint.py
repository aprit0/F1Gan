from torch import nn
from torch.nn.utils import spectral_norm
import numpy as np

class AddDimension(nn.Module):
    def forward(self, x):
        return x.unsqueeze(1)


class SqueezeDimension(nn.Module):
    def forward(self, x):
        return x.squeeze(1)


def create_generator_architecture(input_size, output_size):
    return nn.Sequential(nn.Linear(input_size, 100),
                         nn.LeakyReLU(0.2, inplace=True),
                         AddDimension(),
                         spectral_norm(nn.Conv1d(1, 32, 3, padding=1), n_power_iterations=10),
                         nn.Upsample(200),

                         spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Upsample(400),

                         spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Upsample(800),

                         spectral_norm(nn.Conv1d(32, 1, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),

                         SqueezeDimension(),
                         nn.Linear(800, output_size)
                         )


def create_critic_architecture(calc_flattended):
    return nn.Sequential(AddDimension(),
                         spectral_norm(nn.Conv1d(1, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.MaxPool1d(2),

                         spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.MaxPool1d(2),

                         spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Flatten(),
                         )

'''nn.Linear(calc_flattended, 50),
nn.LeakyReLU(0.2, inplace=True),

nn.Linear(50, 15),
nn.LeakyReLU(0.2, inplace=True),

nn.Linear(15, 1)
)'''


class Generator(nn.Module):
    def __init__(self, input_size=50, output_size=100):
        super().__init__()
        self.main = create_generator_architecture(input_size, output_size)

    def forward(self, input):
        return self.main(input)


class Critic(nn.Module):
    def __init__(self, features=25):
        super().__init__()
        
        self.calc_flattended = int(np.ceil((((features+2) * 0.5 * 0.5 ) + 2)))*32
        self.main = create_critic_architecture(self.calc_flattended)
        self.linear = nn.Sequential(nn.Linear(800, 50),
                         nn.LeakyReLU(0.2, inplace=True),

                         nn.Linear(50, 15),
                         nn.LeakyReLU(0.2, inplace=True),

                         nn.Linear(15, 1)
                         )
    def forward(self, input):
        a = self.main(input)
#         print('C_a', a.shape)
        out = self.linear(a)
        return out
