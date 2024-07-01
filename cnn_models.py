import torch.nn as nn
from torchvision.models import vgg16, resnet101, inception_v3

def get_model(model_name, num_classes, pretrained=True):
    if model_name == 'vgg16':
        model = vgg16(weights="IMAGENET1K_V1")
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        input_size = 224
    
    elif model_name == 'resnet101':
        model = resnet101(weights="IMAGENET1K_V2")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        input_size = 224
    
    elif model_name == 'inceptionv3':
        model = inception_v3(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        input_size = 299
    
    else:
        raise ValueError("Unknown model")

    return model, input_size