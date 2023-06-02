import torch
import torch.nn as nn

class resnet3Dv3(nn.Module):
    def __init__(self, model_type, out_dim):
        super(resnet3Dv3, self).__init__()
        self.resnet = torch.hub.load("facebookresearch/pytorchvideo", model="slow_r50", pretrained=True)
        self.net = nn.Sequential(*list(self.resnet.blocks[0:-1])) 
        self.pool = nn.AvgPool3d((8, 7, 7), stride=(1, 1, 1), padding=(0, 0, 0))
        self.drop = nn.Dropout(0.5)
        self.proj = nn.Linear(2048, out_dim, bias=True)
        self.myfc2 = nn.Sequential(nn.AdaptiveAvgPool3d(output_size=(1,1,1)))
        
    def forward(self, x): 
        x= self.net(x)
        x= self.drop(self.pool(x))
        x = x.permute((0, 2, 3, 4, 1))
        x = self.proj(x)
        x = x.permute((0, 4, 1, 2, 3))
        x = self.myfc2(x)
        x = x.view(x.shape[0], -1)
        
        return x
    
def get_model(model_type, out_dim):
    model = resnet3Dv3(model_type, out_dim)
    return model