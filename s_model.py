import torch.nn as nn

class SequenceModel(nn.Module):
    def __init__(self, input_size=225, output_size=3) -> None:
        super().__init__()
        self.lstm = nn.Sequential(
            nn.LSTM(input_size, 64, 1, batch_first=True),
        )
        self.linear = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x[:,-1,:])

        return x