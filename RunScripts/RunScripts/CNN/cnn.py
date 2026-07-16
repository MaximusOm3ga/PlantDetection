import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,num_classes=10):
        super(CNN,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2)
        )
        self.classifier=nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256*16*16,512),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.1),
            nn.Linear(512,num_classes)

        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x