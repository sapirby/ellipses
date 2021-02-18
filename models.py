import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallNet(nn.Module):
    def __init__(self, input_channels=3):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 11 * 11, 50)
        self.fc_class = nn.Linear(50, 2)
        self.fc_params = nn.Linear(50, 6)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.pool(out)
        out = out.view(-1, 16 * 11 * 11)
        out = F.relu(self.fc1(out))
        x_class = self.fc_class(out)
        x_params = self.fc_params(out)

        return x_class, x_params


if __name__ == "__main__":
    net = SmallNet()
    y_class, y_params = net(torch.randn(1, 3, 50, 50))
    print(y_class.size(), y_params.size())

