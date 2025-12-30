import os
import torch
from torch import nn
from torchvision.models import (
    VGG19_Weights,
    Inception_V3_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    VGG16_Weights,
    vgg19,
    vgg16,
    inception_v3,
    resnet50,
    resnet101
)
from src.cust_models.vgg import VGG

class NeuralStyleTransfer(nn.Module):
    def __init__(self, model_name, pretrained_weights_path=None, pooling='ori', device='cpu'):
        super().__init__()
        self.device = device
        self.model_name = model_name.lower()  # Normalize model name
        self.model = self._create_model(self.model_name, pooling)
        self.freeze()
        
        # load pretrained weights if provided, else use default weights
        if pretrained_weights_path:
            self.load_weights(pretrained_weights_path)
            
        # Move model to specified device
        self.to(device)

    def _create_model(self, model_name, pooling):
        if model_name == "vgg19":
            # Load VGG19 model with pretrained weights
            model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
            model.features = self._replace_pooling_layers(model.features, pooling)
            return model.features

        elif model_name == "vgg16":
            # Load VGG16 model with pretrained weights
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            model.features = self._replace_pooling_layers(model.features, pooling)
            return model.features

        elif model_name == "inception_v3":
            # Load Inception V3 model with pretrained weights
            model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
            model.eval()  # Set to evaluation mode
            # Remove the auxiliary classifier and final classification layer
            model.aux_logits = False
            model.AuxLogits = None
            model = self._replace_pooling_layers(model, pooling)
            return nn.Sequential(*list(model.children())[:-1])

        elif model_name == "resnet50":
            # Load ResNet50 model with pretrained weights
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            # Exclude fully connected and pooling layers
            return nn.Sequential(*list(model.children())[:-2])

        elif model_name == "resnet101":
            # Load ResNet101 model with pretrained weights
            model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            # Exclude fully connected and pooling layers
            return nn.Sequential(*list(model.children())[:-2])

        else:
            raise ValueError("Model name not recognized. Choose from 'vgg19', 'vgg16', 'inception_v3', 'resnet50', or 'resnet101'.")

    def _replace_pooling_layers(self, model, pooling="ori"):
        if pooling == 'avg':
            pooling_layer = nn.AvgPool2d(kernel_size=3, stride=2)
        elif pooling == 'max':
            pooling_layer = nn.MaxPool2d(kernel_size=3, stride=2)
        elif pooling == 'ori':
            return model 
        else:
            raise ValueError("Pooling type not recognized. Choose 'max', 'avg', or 'none'.")

        for name, layer in model.named_children():
            if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                setattr(model, name, pooling_layer)
            elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
                self._replace_pooling_layers(layer, pooling)
        return model

    def forward(self, x, layers):
        features = []
        # Flatten nested lists (for ResNet [[block], [layer]] format)
        flat_layers = []
        for item in layers:
            if isinstance(item, list):
                flat_layers.extend(item)
            else:
                flat_layers.append(item)
        layers = sorted(set(map(int, flat_layers)))  # Sort layers for consistent output
        
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in layers:
                features.append(x)
                
            # Optional early exit if we've collected all required features
            if i > max(layers):
                break
                
        return features
        
    def freeze(self):
        """Freeze all model parameters"""
        for p in self.model.parameters():
            p.requires_grad = False
            
    def load_weights(self, weights_path):
        """Load fine-tuned weights from the specified path."""
        if os.path.isfile(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                # Use strict=False to ignore any mismatched layers
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Successfully loaded weights from {weights_path}.")
            except Exception as e:
                print(f"Error loading weights: {str(e)}")
        else:
            raise FileNotFoundError(f"No weight file found at {weights_path}.")
            
    def get_layer_info(self):
        """Return information about model layers"""
        return {i: str(layer) for i, layer in enumerate(self.model)}