from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.in_channels,
                out_channels=cfg.h1_cnn_channels,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=cfg.h1_cnn_channels,
                out_channels=cfg.h2_cnn_channels,
                kernel_size=3,
                stride=2,
            ),
        )
        self.fc = nn.Sequential(
            nn.Linear(432, cfg.h3_lin_channels),
            nn.ReLU(),
            nn.Linear(cfg.h3_lin_channels, cfg.out_channels),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.size(1) != 1 or x.size(2) != 28 or x.size(3) != 28:
            raise ValueError('Expected each sample to have shape [batch_size, 1, 28, 28]')
        x = self.cnn(x).view(x.size(0), -1)
        return self.fc(x)
