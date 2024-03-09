import torch
import torch.nn as nn
from model.resnet import Bottleneck
from model.stage import Stage

class HRNet(nn.Module):
    def __init__(self, base_channels = 32, out_channels = 17):
        self.base_channels = base_channels
        self.out_channels = out_channels

        super(HRNet, self).__init__()
        self.stem = self.make_stem(inplanes = 3, planes = 64)
        self.layer1 = self.make_layer1(inplanes = 64, planes = 256)
        self.transition1 = self.make_transition1(inplanes = 256, branch1_planes = self.base_channels, branch2_planes = 64)
        self.stage2 = self.make_stage2(base_channels = self.base_channels)
        self.transition2 = self.make_transition2(branch1_inplanes = None, branch2_inplanes = None, branch3_inplanes = 64, 
                                                 branch1_planes = None, branch2_planes = None, branch3_planes = 128)
        self.stage3 = self.make_stage3(base_channels = self.base_channels)
        self.transition3 = self.make_transition3(branch1_inplanes = None, branch2_inplanes = None, branch3_inplanes = None, branch4_inplanes = 128,
                                                 branch1_planes = None, branch2_planes = None, branch3_planes = None, branch4_planes = 256)
        self.stage4 = self.make_stage4(base_channels = self.base_channels)
        self.final_layer = self.make_finallayer(base_channels = self.base_channels, out_channels = self.out_channels)

    def make_stem(self, inplanes = 3, planes = 64, momentum = 0.1):
        return nn.Sequential(nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=2, padding=1, bias=False),
                             nn.BatchNorm2d(num_features=planes, momentum=momentum),
                             nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=2, padding=1, bias=False),
                             nn.BatchNorm2d(num_features=planes, momentum=momentum),
                             nn.ReLU(inplace=True))

    def make_layer1(self, inplanes = 64, planes = 256, stride=1, momentum = 0.1):
        downsample = nn.Sequential(nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(num_features=planes, momentum=momentum),)
        return nn.Sequential(Bottleneck(inplanes=inplanes, planes=inplanes, downsample=downsample),
                             Bottleneck(inplanes=planes, planes=inplanes),
                             Bottleneck(inplanes=planes, planes=inplanes),
                             Bottleneck(inplanes=planes, planes=inplanes))

    def make_transition1(self, inplanes = 256, branch1_planes = 32, branch2_planes = 64, branch1_stride = 1, branch2_stride = 2, momentum = 0.1):
        return nn.ModuleList([nn.Sequential(nn.Conv2d(inplanes, branch1_planes, kernel_size=3, stride=branch1_stride, padding=1, bias=False),
                                            nn.BatchNorm2d(branch1_planes, momentum=momentum),
                                            nn.ReLU(inplace=True)),
                              nn.Sequential(nn.Conv2d(inplanes, branch2_planes, kernel_size=3, stride=branch2_stride, padding=1, bias=False),
                                            nn.BatchNorm2d(branch2_planes, momentum=momentum),
                                            nn.ReLU(inplace=True))])

    def make_stage2(self, base_channels = 32):
        return nn.Sequential(Stage(input_branches=2, output_branches=2, base_channels=base_channels))

    def make_transition2(self, 
                         branch1_inplanes = None, branch2_inplanes = None, branch3_inplanes = 64, 
                         branch1_planes = None, branch2_planes = None, branch3_planes = 128, 
                         branch1_stride = None, branch2_stride = None, branch3_stride = 2, momentum = 0.1):
        return nn.ModuleList([nn.Identity(),
                              nn.Identity(),
                              nn.Sequential(nn.Conv2d(branch3_inplanes, branch3_planes, kernel_size=3, stride=branch3_stride, padding=1, bias=False),
                                            nn.BatchNorm2d(branch3_planes, momentum=momentum),
                                            nn.ReLU(inplace=True))])

    def make_stage3(self, base_channels = 32):
        return nn.Sequential(Stage(input_branches=3, output_branches=3, base_channels=base_channels),
                             Stage(input_branches=3, output_branches=3, base_channels=base_channels),
                             Stage(input_branches=3, output_branches=3, base_channels=base_channels),
                             Stage(input_branches=3, output_branches=3, base_channels=base_channels))

    def make_transition3(self, 
                         branch1_inplanes = None, branch2_inplanes = None, branch3_inplanes = None, branch4_inplanes = 128,
                         branch1_planes = None, branch2_planes = None, branch3_planes = None, branch4_planes = 256, 
                         branch1_stride = None, branch2_stride = None, branch3_stride = None, branch4_stride = 2, momentum = 0.1):
        return nn.ModuleList([nn.Identity(),
                              nn.Identity(),
                              nn.Identity(),
                              nn.Sequential(nn.Conv2d(branch4_inplanes, branch4_planes, kernel_size=3, stride=branch4_stride, padding=1, bias=False),
                                            nn.BatchNorm2d(branch4_planes, momentum=momentum),
                                            nn.ReLU(inplace=True))])
    
    def make_stage4(self, base_channels = 32):
        return nn.Sequential(Stage(input_branches=4, output_branches=4, base_channels=base_channels),
                             Stage(input_branches=4, output_branches=4, base_channels=base_channels),
                             Stage(input_branches=4, output_branches=1, base_channels=base_channels))
    
    def make_finallayer(self, base_channels = 32, out_channels = 17):
        return nn.Conv2d(base_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]
        x = self.stage2(x)
        x = [self.transition2[0](x[0]),
             self.transition2[1](x[1]),
             self.transition2[2](x[1])]
        x = self.stage3(x)
        x = [self.transition3[0](x[0]),
             self.transition3[1](x[1]),
             self.transition3[2](x[2]),
             self.transition3[3](x[2])]
        x = self.stage4(x)
        x = self.final_layer(x[0])
        return x

#example
if __name__ == "__main__": 
    model = HRNet()
    input = torch.randn(1, 3, 256, 192)
    output = model.forward(input)
    print(output.size())