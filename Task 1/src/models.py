import torch.nn as nn
import torch.optim as optim
import torch
import yaml

class Task1Model(nn.Module):
    def __init__(self, dims=[1024, 683, 1], dropouts=[0.5], activation=nn.LeakyReLU(), normalize=False, dtype=torch.float32):
        super().__init__()
        self.activation = activation
        self.dims = dims
        self.dropouts = dropouts
        self.normalize = normalize
        self.dtype = dtype
        self.layers = []

        for i in range(0, len(self.dims)-1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1], dtype=self.dtype))
            if i < (len(self.dims) - 2):
                self.layers.append(self.activation)  
            if self.normalize and i < (len(self.dims) - 2):
                self.layers.append(nn.LayerNorm(self.dims[i+1], dtype=self.dtype))     
            if i < len(self.dropouts):
                self.layers.append(nn.Dropout(self.dropouts[i])) 

        self.layers.append(nn.Sigmoid())
        self.layers_seq = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers_seq(x)
    
def build_model(model_config):
    if type(model_config) == dict:
        try:
            model_type = Task1Model
            dims = model_config['model']['dims']
            dropouts = model_config['model']['dropouts']
            activation = nn.LeakyReLU(model_config['model']['activation']['negative_slope'])
            normalize = model_config['model']['normalize']
            dtype = torch.float32
        except KeyError:
            print("model_config doesn't have the correct structure!")
            return None
        except TypeError:
            print("model_config doesn't have the correct structure!")
            return None
    else:
        with open(model_config, 'r') as f:
            config = yaml.safe_load(f)
            model_type = Task1Model
            dims = config['model']['dims']
            dropouts = config['model']['dropouts']
            activation = nn.LeakyReLU(config['model']['activation']['negative_slope'])
            normalize = config['model']['normalize']
            dtype = torch.float32
    return model_type(dims=dims, dropouts=dropouts, activation=activation, normalize=normalize, dtype=dtype)


test_vector = torch.tensor([1], requires_grad=True, dtype=torch.float32)
test_vector_2 = test_vector**2 + 4
test_vector_2.backward()
print(test_vector.grad)