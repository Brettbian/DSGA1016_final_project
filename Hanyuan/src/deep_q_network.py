"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True))
        # self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.fc1 = nn.Sequential(nn.LazyLinear(256), nn.ReLU())
        self.fc2 = nn.Linear(256, 2)
        # self._create_weights()

    # def _create_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             nn.init.uniform_(m.weight, -0.01, 0.01)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        # output = self.resnet(input)
        output = output.view(output.size(0), -1) #(batch_size, 1000)
        output = self.fc1(output)
        output = self.fc2(output)

        return output
