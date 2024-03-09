from model.resnet import BasicBlock
import torch.nn as nn


class Stage(nn.Module):
    def __init__(self, input_branches, output_branches, base_channels, momentum = 0.1):
        super(Stage, self).__init__()

        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for input_branch_num in range(self.input_branches):
            inplanes = base_channels * (2 ** input_branch_num)
            planes = inplanes
            branch = nn.Sequential(BasicBlock(inplanes=inplanes, planes=planes),
                                   BasicBlock(inplanes=inplanes, planes=planes),
                                   BasicBlock(inplanes=inplanes, planes=planes),
                                   BasicBlock(inplanes=inplanes, planes=planes))
            self.branches.append(branch)
        
        self.fuse_layers = nn.ModuleList()
        for output_branch_num in range(self.output_branches):

            fuse_layer = nn.ModuleList()
            for input_branch_num in range(self.input_branches):
                if output_branch_num == input_branch_num:#Same channels, do nothing.
                    fuse_layer.append(nn.Identity())

                if output_branch_num < input_branch_num:#upsampling.
                    upsampling_inplanes = base_channels * (2 ** input_branch_num)
                    upsampling_planes = base_channels * (2 ** output_branch_num)
                    upsampling_scale_factor = 2.0 ** (input_branch_num - output_branch_num)

                    fuse_layer.append(nn.Sequential(nn.Conv2d(upsampling_inplanes, upsampling_planes, kernel_size=1, stride=1, bias=False),
                                                    nn.BatchNorm2d(upsampling_planes, momentum=momentum),
                                                    nn.Upsample(scale_factor=upsampling_scale_factor, mode='nearest')))

                if output_branch_num > input_branch_num:#downsampling.
                    downsampling_inplanes = base_channels * (2 ** input_branch_num)
                    downsampling_planes = base_channels * (2 ** output_branch_num)

                    ops = []
                    for k in range(output_branch_num - input_branch_num - 1):
                        ops.append(nn.Sequential(nn.Conv2d(downsampling_inplanes, downsampling_inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                                                 nn.BatchNorm2d(downsampling_inplanes, momentum=momentum),
                                                 nn.ReLU(inplace=True)))
                        
                    ops.append(nn.Sequential(nn.Conv2d(downsampling_inplanes, downsampling_planes, kernel_size=3, stride=2, padding=1, bias=False),
                                             nn.BatchNorm2d(downsampling_planes, momentum=momentum)))
                    
                    fuse_layer.append(nn.Sequential(*ops))
            self.fuse_layers.append(fuse_layer)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_list):
        x_list = [branch(x) for branch, x in zip(self.branches, x_list)]

        x_fused_list = []
        for output_branch_num in range(self.output_branches):
            x_sum = 0.0
            for input_branch_num in range(self.input_branches):
                x_sum += self.fuse_layers[output_branch_num][input_branch_num](x_list[input_branch_num])
            x_fused_list.append(self.relu(x_sum))

        return x_fused_list