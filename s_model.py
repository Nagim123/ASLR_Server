import torch.nn as nn

class GruModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes) -> None:
        super(GruModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_size // 2, hidden_size // 4),
#             nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
#         out = self.fc(out[:, -1, :])
        out = self.fc(torch.mean(out, 1))
        return out
