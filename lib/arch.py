
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from collections import OrderedDict
import functools
from torch.optim import lr_scheduler
from torch.nn.utils import spectral_norm

# n_input and ngf influence the generator architecture in code. 
# n_imput is the length of the input vector, 
# ngf relates to the size of the feature maps that are propagated through the generator, 
# the output channel are always 3

def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False):
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def upsample_block(in_planes, out_planes, dropout=0.0):
    block = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_planes, out_planes, kernel_size= 3, stride= 1, padding= 1, bias= False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout)
            ]

    block = nn.Sequential(*block)
    return block

class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block = conv_block + [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block = conv_block + [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block = conv_block + [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(inplace=False)]
        
        if dropout > 0.0:
            conv_block = conv_block + [nn.Dropout(dropout)]
        p = 0
        if padding_type == 'reflect':
            conv_block = conv_block + [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block = conv_block + [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block = conv_block + [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)  
        return out

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()

        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.1, inplace=False))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out= self.model(x)
        return out

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace= False)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        out = self.model(x)
        out = torch.cat((out, skip_input), 1)
        return out

class GeneratorResNet(nn.Module):

    def __init__(self, config):
        super(GeneratorResNet, self).__init__()

        n_input = config.N_INPUT
        ngf = config.NGF
        self.ngf = ngf
        self.norm_layer = config.NORM_LAYER
        self.dropout = config.G_DROPOUT
        self.padding_type = config.PADDING_TYPE
        n_blocks = config.N_BLOCKS
        assert(n_blocks >= 0)

        if type(self.norm_layer) == functools.partial:
            self.use_bias = self.norm_layer.func == nn.InstanceNorm2d
        else:
            self.use_bias = self.norm_layer == nn.InstanceNorm2d

        self.fc = nn.Sequential(
                               nn.Linear(n_input, ngf * 4 * 4 * 4, bias=False),
                                nn.ReLU(inplace=False),
                                nn.Dropout(self.dropout)
                               )

        self.start = nn.Sequential(nn.ReflectionPad2d(3),
                                       nn.Conv2d(ngf * 4, ngf * 4, kernel_size=7, padding=0, bias=self.use_bias),
                                       self.norm_layer(ngf * 4),       
                                       nn.ReLU(inplace=False))

        self.upsample1 = upsample_block(ngf * 4, ngf * 2, dropout=self.dropout)         
        self.upsample2 = upsample_block(ngf * 2, ngf * 2, dropout=self.dropout)      
        self.upsample3 = upsample_block(ngf * 2, ngf, dropout=self.dropout)     
        self.upsample4 = upsample_block(ngf, ngf, dropout=self.dropout)    
        self.block = self._build_blocks(ngf, n_blocks)  

        self.finish = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(ngf, 3, kernel_size=7, padding=0),   
                                        nn.Tanh())

    def _build_blocks(self, channel, n_blocks):
        blocks = []
        for i in range(n_blocks):       
            blocks = blocks + [ResnetBlock(channel,
                                    padding_type=self.padding_type,
                                    norm_layer=self.norm_layer,
                                    dropout=self.dropout,
                                    use_bias=self.use_bias)
                                    ]
        blocks = nn.Sequential(*blocks)          
        return blocks

    def forward(self, word_vectors):           
        out = self.fc(word_vectors)      
        out = out.view(-1, self.ngf * 4, 4, 4) 
        out = self.start(out) 
        out = self.upsample1(out)  
        out = self.upsample2(out)  
        out = self.upsample3(out)  
        out = self.upsample4(out) 
        out = self.block(out) 
        out = self.finish(out)   

        return out

class GeneratorRefinerUNet(nn.Module):
    def __init__(self, config):
        super(GeneratorRefinerUNet, self).__init__()

        n_channels = config.N_CHANNELS
        ngf = config.NG_REF_F
        self.norm_layer = config.NORM_LAYER
        dropout = config.G_DROPOUT

        self.down1 = UNetDown(n_channels, ngf * 2, normalize=False)
        self.down2 = UNetDown(ngf * 2, ngf * 4)
        self.down3 = UNetDown(ngf * 4, ngf * 8)
        self.down4 = UNetDown(ngf * 8, ngf * 8, dropout=dropout)

        self.up1 = UNetUp(ngf * 8, ngf * 8, dropout=dropout)
        self.up2 = UNetUp(ngf * 8 + ngf * 8, ngf * 4, dropout=dropout)
        self.up3 = UNetUp(ngf * 4 + ngf * 4, ngf * 4, dropout=dropout)

        self.finish = nn.Sequential(nn.Conv2d(342, 3, kernel_size=1, padding=0),
                                    nn.Upsample(size= (128,128)),    
                                    nn.Tanh())
        self.dense = self._build_blocks(n_blocks= 1,dropout= dropout)

    
    def _build_blocks(self, n_blocks, dropout):
        blocks = []
        for i in range(n_blocks):      
            blocks = blocks + [DenseNet(drop_rate= dropout,
                                block_config=(16, 16, 16),
                                num_init_features=24)
                                ]

        blocks = nn.Sequential(*blocks)                  # -> ngf x H x W
        return blocks

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        dense = self.dense(u3)
        final= self.finish(dense)
        return final

class GeneratorRefinerUNet2(nn.Module):
    def __init__(self, config):
        super(GeneratorRefinerUNet2, self).__init__()

        n_channels = config.N_CHANNELS
        ngf = config.NG_REF_F
        self.norm_layer = config.NORM_LAYER
        dropout = config.G_DROPOUT

        self.down1 = UNetDown(n_channels, ngf, normalize=False)
        self.down2 = UNetDown(ngf, ngf * 2)
        self.down3 = UNetDown(ngf * 2, ngf * 4)
        self.down4 = UNetDown(ngf * 4, ngf * 8, dropout=dropout)
        self.down5 = UNetDown(ngf * 8, ngf * 8, dropout=dropout)

        self.up1 = UNetUp(ngf * 8, ngf * 8, dropout=dropout)
        self.up2 = UNetUp(ngf * 8 + ngf * 8, ngf * 4, dropout=dropout)
        self.up3 = UNetUp(ngf * 4 + ngf * 4, ngf * 2, dropout=dropout)
        self.up4 = UNetUp(ngf * 2 + ngf * 2, ngf * 2, dropout=dropout)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(ngf * 2 + ngf, ngf, 4, padding=1),
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(ngf, n_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        final= self.final(u4)
        return final

class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=False)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=False)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=False))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))

class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0, avgpool_size=8):
        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = avgpool_size
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(384, num_init_features, kernel_size=2, stride=1, padding=1, bias=False)),
        ]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(num_input_features=num_features,
                                    num_output_features=int(num_features
                                                            * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

    def forward(self, x):     
        features = self.features(x)
        out = F.relu(features, inplace= False)
        return out

class DiscriminatorStack(nn.Module):
    def __init__(self, config):
        super(DiscriminatorStack, self).__init__()
        ndf = config.NDF
        ngf = config.NGF
        fc_in = config.N_INPUT
        fc_out = config.IMAGE_WIDTH_FIRST * config.IMAGE_HEIGHT_FIRST
        n_channels = config.N_CHANNELS + 1   
        self.ndf = ndf
        self.ngf = ngf
        self.out_channels = config.OUT_CHANNELS
        dropout = config.D_DROPOUT
        batch_size = config.BATCH_SIZE

        self.fc = nn.Sequential(
                             nn.Linear(fc_in, fc_out, bias=False),
                            nn.ReLU(inplace=False),
                            nn.Dropout(dropout)
                            )

        self.conv = [  
            nn.Conv2d(n_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),          
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),   
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(ndf * 4, self.out_channels, kernel_size=4, stride=2, padding=1, bias=False),     
            nn.Dropout2d(dropout)
        ]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, image, word_vectors):
        b, _, h, w = image.size()
        wv_out = self.fc(word_vectors)
        wv_out = wv_out.view(b, 1, h, w)     
        stacked = torch.cat((image, wv_out), dim=1)
        out = self.conv(stacked)
        return out

class DiscriminatorDecider(nn.Module):
    def __init__(self, config):
        super(DiscriminatorDecider, self).__init__()
        ndf = config.ND_DEC_F
        self.ndf = ndf
        n_channels = config.N_CHANNELS
        self.out_channels = config.OUT_CHANNELS
        dropout = config.D_DROPOUT
        batch_size = config.BATCH_SIZE

        self.conv = [                                                      
            nn.Conv2d(n_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),           
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),   
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),     
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(ndf * 8, self.out_channels, kernel_size=4, stride=2, padding=1, bias=False),     
            nn.Dropout2d(dropout)
        ]
                
        self.conv1= nn.Conv2d(n_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False)           
        self.leaky1= nn.LeakyReLU(0.2, inplace=False)
        self.conv2= nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)   
        self.norm1= nn.BatchNorm2d(ndf * 2)
        self.leaky2= nn.LeakyReLU(0.1, inplace=False)
        self.conv3=  nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm2=  nn.BatchNorm2d(ndf * 4)
        self.leaky3= nn.LeakyReLU(0.1, inplace=False)
        self.conv4= nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm3= nn.BatchNorm2d(ndf * 8)
        self.leaky4= nn.ReLU(inplace=False)
        self.conv5= nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False)       ## ndf * 8 x H/32 x W/32
        self.norm4= nn.BatchNorm2d(ndf * 8)
        self.relu= nn.ReLU(inplace=False)
        self.conv6= nn.Conv2d(ndf * 8, self.out_channels, kernel_size=4, stride=2, padding=1, bias=False) 



        self.conv = nn.Sequential(*self.conv)

    def forward(self, image):
        out= self.conv1(image) 
        out= self.leaky1(out)
        out= self.conv2(out) 
        out= self.norm1(out) 
        out= self.leaky2(out)
        out= self.conv3(out)
        out= self.norm2(out) 
        out= self.leaky3(out) 
        out= self.conv4(out) 
        out= self.norm3(out)
        out= self.leaky4(out) 
        out= self.conv5(out)
        out= self.norm4(out) 
        out= self.relu(out)
        out= self.conv5(out)     
#        out = self.conv(image).clone()   
        return out

class DiscriminatorDecider2(nn.Module):
    def __init__(self, config):
        super(DiscriminatorDecider2, self).__init__()
        ndf = config.ND_DEC_F
        self.ndf = ndf
        n_channels = config.N_CHANNELS
        self.out_channels = config.OUT_CHANNELS
        dropout = config.D_DROPOUT
        batch_size = config.BATCH_SIZE

        self.conv = [                                                 
            nn.Conv2d(n_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),          
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),     
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(ndf * 4, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),    
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),       
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(ndf * 8, self.out_channels, kernel_size=4, stride=2, padding=1, bias=False),       
            nn.Dropout2d(dropout)
        ]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, image):       
        out = self.conv(image)
        return out

if __name__ == '__main__':
    from config import Config
    config = Config()
    G = GeneratorResNet(config)
    image = torch.Tensor(2, 4096)
    G(image)