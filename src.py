import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T


class AutoEncoder(nn.Module):
    
    def __init__(self, output_classes: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(2, return_indices=True),
        )
        
        self.encoder_linear = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(4480, 1028),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1028, 512),
            nn.ReLU(),
            nn.Linear(512, output_classes),
        )
        
        self.decoder_linear = nn.Sequential(
            nn.Linear(output_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 1028),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1028, 4480),
            nn.Unflatten(1, (64, 10, 7)),
        )
        
        self.unpool = nn.MaxUnpool2d(2)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, 2),
        )
                    
    def encode(self, x):
        x, indices = self.encoder(x)
        x = self.encoder_linear(x)
        return x, indices
        
    def decode(self, x, indices):
        x = self.decoder_linear(x)
        x = self.unpool(x, indices, output_size=(x.shape[0], 64, 21, 14))
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        encoded, indices = self.encode(x)
        encoded = F.softmax(encoded, dim=1)
        decoded = self.decode(encoded, indices)
        return encoded, decoded
    
    
class AEFeatureExtractor:
    
    _model: nn.Module = AutoEncoder(24)
    _transform_val = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def __init__(self, path: str):
        self._model.load_state_dict(torch.load(path))
        self._model.eval()
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) > 3:
            raise Exception()
        image = self._transform_val(image)[None, :]
        outputs, encoded = self._model(image)
        return outputs.detach().numpy().squeeze(), encoded.detach().numpy().squeeze()
    