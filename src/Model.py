import torch.nn as nn

class Net(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor # output size [batch_size, 512, 1 1]
        self.classifier = nn.Sequential(*[nn.Linear(512, 3),
                                          nn.Softmax(dim=-1)])
        
    def forward(self, x):
        x = self.feature_extractor(x).view(-1, 512)
        x = self.classifier(x)
        return x