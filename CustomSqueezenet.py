from torchvision.models import squeezenet1_0
import torch.nn as nn

class CustomSqueezeNet(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomSqueezeNet, self).__init__()
        
        # Load the pre-trained SqueezeNet model
        squeezenet = squeezenet1_0(pretrained=True)
        
        # Copy the features from the pre-trained model
        self.features = squeezenet.features
        
        # Replace the classifier with a custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Softmax(dim=1)
        )
        
        # Initialize the weights for the new layers
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
