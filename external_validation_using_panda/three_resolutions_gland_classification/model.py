import torchvision
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=2, num_intermediate_features=64):
        super().__init__()
        
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.num_intermediate_features = num_intermediate_features
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.in_features, num_intermediate_features, bias=True)
        self.fc2 = nn.Linear(num_intermediate_features, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        feature_vec = self.resnet(x)
        feature_vec = self.dropout(feature_vec)
        class_score_vec = self.fc2(feature_vec)
        
        return feature_vec, class_score_vec
    
    
class Model(nn.Module):
    """
    Creates a multi-resolution gland classification model. There is one ResNet18
    as feature extractor for each resolution. Then, extracted feature vectors are 
    summed up and fed to a linear classifier.

    Args:
        pretrained (bool): Flag determining if feature extractors are 
            pretrained (True) or not (False). Default: False
        num_classes (int): The number of classes in the classification task. Default: 2
        num_intermediate_features (int): Size of the output vector for 
            feature extractors. Default: 64
    """

    def __init__(self, pretrained=False, num_classes=2, num_intermediate_features=64):
        super().__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.num_intermediate_features = num_intermediate_features
        self.resnet_high = ResNet(pretrained, num_classes, num_intermediate_features)
        self.resnet_medium = ResNet(pretrained, num_classes, num_intermediate_features)
        self.resnet_low = ResNet(pretrained, num_classes, num_intermediate_features)
        self.fc3 = nn.Linear(num_intermediate_features, 10)
        self.fc4 = nn.Linear(10, num_classes)
        self.dropout_high = nn.Dropout(0.5)
        self.dropout_medium = nn.Dropout(0.5)
        self.dropout_low = nn.Dropout(0.5)
        self.dropout_result = nn.Dropout(0.5)
        self.dropout_fc3 = nn.Dropout(0.5)
        
    def forward(self, x_high, x_medium, x_low):
        feature_vec_high, class_score_vec_high = self.resnet_high(x_high)
        feature_vec_high = self.dropout_high(feature_vec_high)

        feature_vec_medium, class_score_vec_medium = self.resnet_medium(x_medium)
        feature_vec_medium = self.dropout_medium(feature_vec_medium)

        feature_vec_low, class_score_vec_low = self.resnet_low(x_low)
        feature_vec_low = self.dropout_low(feature_vec_low)

        feature_vec = feature_vec_high + feature_vec_medium + feature_vec_low
        feature_vec = self.dropout_result(feature_vec)
        feature_vec = self.fc3(feature_vec)
        feature_vec = self.dropout_fc3(feature_vec)
        class_score_vec = self.fc4(feature_vec)
        
        return class_score_vec_high, class_score_vec_medium, class_score_vec_low, class_score_vec

        