import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention_branch(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(384, 128, 1))
        self.layers.append(nn.Conv2d(128, 32, (3, 3), stride=1, padding=(4, 4), bias=False, dilation=(4, 4)))
        self.layers.append(nn.BatchNorm2d(32, eps=1e-03))

        self.layers_final = nn.ModuleList()

        self.layers_final.append(nn.Dropout2d(0.1))
        self.layers_final.append(nn.Conv2d(32, 5, (1, 1), stride=1, padding=(0, 0), bias=True))

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.linear1 = nn.Linear(4500, 2028)
        self.linear2 = nn.Linear(2028, 1152)
    
    def forward(self, input):
        output = input 

        for layer in self.layers:
            output = layer(output)

        output = F.relu(output)

        for layer in self.layers_final:
            output = layer(output)

        output = F.softmax(output, dim=1)
        output = self.maxpool(output)
        output = output.view(-1, 4500)
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)
        output = output.view(-1, 1, 288, 4)
        output = F.sigmoid(output)
        output = output.expand(-1, 800, 288, 4)
        output = output.permute(0, 3, 2, 1)

        return output


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.1, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.1, 2))
            self.layers.append(non_bottleneck_1d(128, 0.1, 4))
            self.layers.append(non_bottleneck_1d(128, 0.1, 8))
            self.layers.append(non_bottleneck_1d(128, 0.1, 16))

        # only for encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        x = self.initial_block(input)
        x1 = self.layers[0](x)
        
        x = self.layers[1](x1)
        x = self.layers[2](x)
        x = self.layers[3](x)
        x = self.layers[4](x)
        x2 = self.layers[5](x)

        x = self.layers[6](x2)
        x = self.layers[7](x)
        x = self.layers[8](x)
        x = self.layers[9](x)
        x3 = self.layers[10](x)

        x = self.layers[11](x3)
        x = self.layers[12](x)
        x = self.layers[13](x)
        x4 = self.layers[14](x)

        return x1, x2, x3, x4


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3, track_running_stats=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


class Lane_exist(nn.Module):
    def __init__(self, num_output):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(128, 32, (3, 3), stride=1, padding=(4, 4), bias=False, dilation=(4, 4)))
        self.layers.append(nn.BatchNorm2d(32, eps=1e-03))

        self.layers_final = nn.ModuleList()

        self.layers_final.append(nn.Dropout2d(0.1))
        self.layers_final.append(nn.Conv2d(32, 5, (1, 1), stride=1, padding=(0, 0), bias=True))

        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.linear1 = nn.Linear(4500, 128)
        self.linear2 = nn.Linear(128, num_output)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = F.relu(output)

        for layer in self.layers_final:
            output = layer(output)

        output = F.softmax(output, dim=1)
        output = self.maxpool(output)
        output = output.view(-1, 4500)
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)
        output = F.sigmoid(output)

        return output


class mymodel(nn.Module):
    def __init__(self, num_classes, encoder=None):  # use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)
        self.lane_exist = Lane_exist(4)  # num_output
        self.input_mean = [103.939, 116.779, 123.68]  # [0, 0, 0]
        self.input_std = [1, 1, 1]

        self.att_branch = Attention_branch()


    def train(self, mode=True):
        super(mymodel, self).train(mode)

    def forward(self, input):
        x1, x2, x3, x4 = self.encoder(input)

        x1 = torch.nn.functional.interpolate(x1, scale_factor=0.5, mode='bilinear')
        x2 = torch.nn.functional.interpolate(x2, scale_factor=0.5, mode='bilinear')
        
        cat_x = torch.cat([x1, x2, x3, x4], dim=1)
        
        return self.decoder.forward(x4), self.att_branch(cat_x), self.lane_exist(x4)

# if __name__ == "__main__":
#     test = torch.rand(4, 3, 288, 800)

#     net = ERFNet(num_classes=5)
#     net(test)