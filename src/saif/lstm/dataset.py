import torch
from torch.utils.data import Dataset

class SeismicDataset(Dataset):
    def __init__(self, df, features, target, seq_length):
        '''
        df: pd.DataFrame
        features: (list of str) Feature columns in dataframe
        target: (str) Target column in dataframe
        seq_length: (int) Sequence length
        '''
        self.features = features
        self.target = target
        self.seq_length = seq_length
        self.X = torch.tensor(df[features].values).float()
        self.y = torch.tensor(df[target].values).float()

    def __len__(self):
        '''
        Checks for length of the dataset
        '''
        return self.X.shape[0]

    def __getitem__(self, i):
        '''
        Returns rows (i - seq_leng) to i upon querying i^th element of the dataset.
        If i is near the beginning of the dataset, we pad by repeating the first row as many times as needed to make the output have seq_length rows.
        '''
        if i >= self.seq_length - 1:
            i_start = i - self.seq_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.seq_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)
        return x, self.y[i]
