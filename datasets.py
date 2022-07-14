from typing import Sequence

from torch.utils.data import Dataset
import numpy as np



class Sines(Dataset):

    def __init__(self, frequency_range: Sequence[float], amplitude_range: Sequence[float],
                 n_series: int = 200, datapoints: int = 100, seed: int = None):
        """
        Pytorch Dataset to produce sines.

        y = A * sin(B * x)

        :param frequency_range: range of A
        :param amplitude_range: range of B
        :param n_series: number of sines in your dataset
        :param datapoints: length of each sample
        :param seed: random seed
        """
        self.n_series = n_series
        self.datapoints = datapoints
        self.seed = seed
        self.frequency_range = frequency_range
        self.amplitude_range = amplitude_range
        self.dataset = self._generate_sines()

    def __len__(self):
        return self.n_series

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _generate_sines(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        x = np.linspace(start=0, stop=2 * np.pi, num=self.datapoints)
        low_freq, up_freq = self.frequency_range[0], self.frequency_range[1]
        low_amp, up_amp = self.amplitude_range[0], self.amplitude_range[1]

        freq_vector = (up_freq - low_freq) * np.random.rand(self.n_series, 1) + low_freq
        ampl_vector = (up_amp - low_amp) * np.random.rand(self.n_series, 1) + low_amp

        return ampl_vector * np.sin(freq_vector * x)
class Load(Dataset):

    def __init__(self, file_path, lookback):
        self.lookback = lookback
        self.file_path = file_path
        self.dataset = self._load_numpy()
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        #dataX[i - T] = df[i - T:i, :]
        #print('DATALOAD: ', idx, self.dataset[idx].shape, self.dataset.shape)
        
        #print('Data', data.shape)
        return self.dataset[idx]
    
    def _load_numpy(self):
        arr = np.load(self.file_path + '.npy')
        '''arr = arr.T
        data = [0] * (arr.shape[0] - self.lookback)
        for idx in range(self.lookback, arr.shape[0]):
            data[idx - self.lookback] = arr[idx - self.lookback:idx,:]'''
        return arr#np.array(data)