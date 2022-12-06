
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.nn.utils import spectral_norm

def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False, use_spectral=False):
    if use_spectral:
        return spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

def upsample_block(in_planes, out_planes, dropout=0.0, use_spectral=False):
    # Upsample the spatial size by a factor of 2
    block = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(in_planes, out_planes, use_spectral=use_spectral),
            nn.BatchNorm2d(out_planes),
#            nn.ReLU(True),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout)
            ]

    block = nn.Sequential(*block)
    return block

class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, dropout, use_bias, use_spectral):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, dropout, use_bias, use_spectral)

    def build_conv_block(self, dim, padding_type, norm_layer, dropout, use_bias, use_spectral):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
#            conv_block += [nn.ReflectionPad2d(1)]
            conv_block = conv_block + [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
#            conv_block += [nn.ReplicationPad2d(1)]
            conv_block = conv_block + [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if use_spectral:
#            conv_block += [spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)), norm_layer(dim), nn.ReLU(True)]
            conv_block = conv_block + [spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)), norm_layer(dim), nn.ReLU(inplace=False)]
        else:
#            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
            conv_block = conv_block + [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(inplace=False)]
        if dropout > 0.0:
#            conv_block += [nn.Dropout(dropout)]
            conv_block = conv_block + [nn.Dropout(dropout)]
        p = 0
        if padding_type == 'reflect':
#            conv_block += [nn.ReflectionPad2d(1)]
            conv_block = conv_block + [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
#            conv_block += [nn.ReplicationPad2d(1)]
            conv_block = conv_block + [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        if use_spectral:
#            conv_block += [spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)), norm_layer(dim)]
            conv_block = conv_block + [spectral_norm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)), norm_layer(dim)]
        else:
#            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
            conv_block = conv_block + [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)  # add skip connections
        print ("Resnet : ", out.size())
        return out

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, use_spectral=True):
        super(UNetDown, self).__init__()
        if use_spectral:
            layers = [spectral_norm(nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False))]
        else:
            layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
#        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        print ("UNet D: ", self.model(x).size())
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
#            nn.ReLU(inplace=True),
            nn.ReLU(inplace= False)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        print ("UNet U: ", x.size())
        return x

class GeneratorResNet(nn.Module):

    def __init__(self, config):
        super(GeneratorResNet, self).__init__()

        n_input = config.N_INPUT
        ngf = config.NGF
        self.ngf = ngf
        self.norm_layer = config.NORM_LAYER
        self.use_spectral = config.USE_SPECTRAL_NORM
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
#                               nn.ReLU(True),
                                nn.ReLU(inplace=False),
                               nn.Dropout(self.dropout)
                               )

        if self.use_spectral:
            self.start = nn.Sequential(nn.ReflectionPad2d(3),
                                       spectral_norm(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=7, padding=0, bias=self.use_bias)),
                                       self.norm_layer(ngf * 4),
#                                       nn.ReLU(True))             # -> ngf x H x W
                                       nn.ReLU(inplace=False))
        else:
            self.start = nn.Sequential(nn.ReflectionPad2d(3),
                                       nn.Conv2d(ngf * 4, ngf * 4, kernel_size=7, padding=0, bias=self.use_bias),
                                       self.norm_layer(ngf * 4),
#                                       nn.ReLU(True))             # -> ngf x H x W
                                       nn.ReLU(inplace=False))

        # self.block1 = self._build_blocks(ngf, n_blocks)
        self.upsample1 = upsample_block(ngf * 4, ngf * 2, dropout=self.dropout)           # ngf x H x W -> ngf/2 x 2H x 2W
        self.upsample2 = upsample_block(ngf * 2, ngf * 2, dropout=self.dropout)      # -> ngf/4 x 4H x 4W
        self.upsample3 = upsample_block(ngf * 2, ngf, dropout=self.dropout)      # -> ngf/8 x 8H x 8W
        self.upsample4 = upsample_block(ngf, ngf, dropout=self.dropout)     # -> ngf/16 x 16H x 16W
        self.block = self._build_blocks(ngf, n_blocks)
        # self.upsample5 = upsample_block(ngf // 16, ngf // 32)     # -> ngf/32 x 32H x 32W

        if self.use_spectral:
            self.finish = nn.Sequential(nn.ReflectionPad2d(3),
                                        spectral_norm(nn.Conv2d(ngf, 3, kernel_size=7, padding=0)),    # 3 x 16H x 16W
                                        nn.Tanh())
        else:
            self.finish = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(ngf, 3, kernel_size=7, padding=0),    # 3 x 16H x 16W
                                        nn.Tanh())

    def _build_blocks(self, channel, n_blocks):
        blocks = []
        for i in range(n_blocks):       # add ResNet blocks
#            blocks += [ResnetBlock(channel,
            blocks = blocks + [ResnetBlock(channel,
                                    padding_type=self.padding_type,
                                    norm_layer=self.norm_layer,
                                    dropout=self.dropout,
                                    use_bias=self.use_bias,
                                    use_spectral=self.use_spectral)
                                    ]
        blocks = nn.Sequential(*blocks)                  # -> ngf x H x W
        return blocks

    def forward(self, word_vectors):
        out = self.fc(word_vectors)
        print ("G Resnet: 1", out.size())        
        out = out.view(-1, self.ngf * 4, 4, 4)
        print ("G Resnet: 2", out.size())   
        out = self.start(out)
        print ("G Resnet: 3", out.size())   
        out = self.upsample1(out)
        print ("G Resnet: 4", out.size())   
        out = self.upsample2(out)
        print ("G Resnet: 5", out.size())   
        out = self.upsample3(out)
        print ("G Resnet: 6", out.size())   
        out = self.upsample4(out)
        print ("G Resnet: 7", out.size())   
        out = self.block(out)
        print ("G Resnet: 8", out.size())   
        out = self.finish(out)
        print ("G Resnet: 9", out.size())   

        return out

class GeneratorRefinerUNet(nn.Module):
    def __init__(self, config):
        super(GeneratorRefinerUNet, self).__init__()

        n_channels = config.N_CHANNELS
        ngf = config.NG_REF_F
        self.norm_layer = config.NORM_LAYER
        self.use_spectral = config.USE_SPECTRAL_NORM
        dropout = config.G_DROPOUT

        self.down1 = UNetDown(n_channels, ngf * 2, normalize=False, use_spectral=self.use_spectral)
        self.down2 = UNetDown(ngf * 2, ngf * 4, use_spectral=self.use_spectral)
        self.down3 = UNetDown(ngf * 4, ngf * 8, use_spectral=self.use_spectral)
        self.down4 = UNetDown(ngf * 8, ngf * 8, dropout=dropout, use_spectral=self.use_spectral)

        self.up1 = UNetUp(ngf * 8, ngf * 8, dropout=dropout)
        self.up2 = UNetUp(ngf * 8 + ngf * 8, ngf * 4, dropout=dropout)
        self.up3 = UNetUp(ngf * 4 + ngf * 4, ngf * 4, dropout=dropout)

        self.final = nn.Sequential(
                                  nn.Upsample(scale_factor=2),
                                  nn.ZeroPad2d((1, 0, 1, 0)),
                                  nn.Conv2d(ngf * 4 + ngf * 2, ngf * 2, 4, padding=1),
                                  nn.Upsample(scale_factor=2),
                                  nn.ZeroPad2d((1, 0, 1, 0)),
                                  nn.Conv2d(ngf * 2, n_channels, 4, padding=1),
                                  nn.Tanh()
                                  )

    def forward(self, x):
        print ("RefU 1: 1", x.size())
        d1 = self.down1(x)
        print ("RefU 1: 2", d1.size())
        d2 = self.down2(d1)
        print ("RefU 1: 3", d2.size())
        d3 = self.down3(d2)
        print ("RefU 1: 4", d3.size())
        d4 = self.down4(d3)
        print ("RefU 1: 5", d4.size())
        u1 = self.up1(d4, d3)
        print ("RefU 1: 6", u1.size())
        u2 = self.up2(u1, d2)
        print ("RefU 1: 7", u2.size())
        u3 = self.up3(u2, d1)
        print ("RefU 1: 8", u3.size())
        final= self.final(u3)
        print ("RefU 1: 9", final.size())        
        return final

class GeneratorRefinerUNet2(nn.Module):
    def __init__(self, config):
        super(GeneratorRefinerUNet2, self).__init__()

        n_channels = config.N_CHANNELS
        ngf = config.NG_REF_F
        self.norm_layer = config.NORM_LAYER
        self.use_spectral = config.USE_SPECTRAL_NORM
        dropout = config.G_DROPOUT

        self.down1 = UNetDown(n_channels, ngf, normalize=False, use_spectral=self.use_spectral)
        self.down2 = UNetDown(ngf, ngf * 2, use_spectral=self.use_spectral)
        self.down3 = UNetDown(ngf * 2, ngf * 4, use_spectral=self.use_spectral)
        self.down4 = UNetDown(ngf * 4, ngf * 8, dropout=dropout, use_spectral=self.use_spectral)
        self.down5 = UNetDown(ngf * 8, ngf * 8, dropout=dropout, use_spectral=self.use_spectral)

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
        print ("RefU 2: 1", x.size())
        d1 = self.down1(x)
        print ("RefU 2: 2",d1.size())
        d2 = self.down2(d1)
        print ("RefU 2: 3",d2.size())
        d3 = self.down3(d2)
        print ("RefU 2: 4",d3.size())
        d4 = self.down4(d3)
        print ("RefU 2: 5",d4.size())
        d5 = self.down5(d4)
        print ("RefU 2: 6",d5.size())
        u1 = self.up1(d5, d4)
        print ("RefU 2: 7",u1.size())
        u2 = self.up2(u1, d3)
        print ("RefU 2: 8",u2.size())
        u3 = self.up3(u2, d2)
        print ("RefU 2: 9",u3.size())
        u4 = self.up4(u3, d1)
        print ("RefU 2: 10",u4.size())
        final= self.final(u4)
        print ("RefU 2: 10",final.size())
        return final

class DiscriminatorStack(nn.Module):
    def __init__(self, config):
        super(DiscriminatorStack, self).__init__()
        ndf = config.NDF
        ngf = config.NGF
        fc_in = config.N_INPUT
        fc_out = config.IMAGE_WIDTH_FIRST * config.IMAGE_HEIGHT_FIRST
        n_channels = config.N_CHANNELS + 1   ## Stitching images and word vectors
        self.ndf = ndf
        self.ngf = ngf
        self.out_channels = config.OUT_CHANNELS
        use_spectral = config.USE_SPECTRAL_NORM
        dropout = config.D_DROPOUT
        self.minibatch_discrimination = config.MINIBATCH_DISCRIMINATION
        batch_size = config.BATCH_SIZE

        ## No Batch Normalization
        self.fc = nn.Sequential(
                            nn.Linear(fc_in, fc_out, bias=False),
#                            nn.ReLU(True),
                            nn.ReLU(inplace=False),
                            nn.Dropout(dropout)
                            )

        if use_spectral:
            self.conv = [                                                       ## 4 x H x W
                spectral_norm(nn.Conv2d(n_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False)),           ## ndf x H/2 x W/2
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)),     ## ndf * 2 x H/4 x W/4
                nn.BatchNorm2d(ndf * 2),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)), ## ndf * 4 x H/8 x W/8
                nn.BatchNorm2d(ndf * 2),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)), ## ndf * 8 x H/16 x W/16
                nn.BatchNorm2d(ndf * 4),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                spectral_norm(nn.Conv2d(ndf * 4, self.out_channels, kernel_size=4, stride=2, padding=1, bias=False)),       ## ndf * 8 x H/32 x W/32
                # nn.BatchNorm2d(self.out_channels),
                # nn.LeakyReLU(0.2, inplace=True),
                # nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1, bias=False),       ## 1 x H/64 x W/64
                nn.Dropout2d(dropout)
            ]
        else:
            self.conv = [                                                       ## 4 x H x W
                nn.Conv2d(n_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),           ## ndf x H/2 x W/2
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),     ## ndf * 2 x H/4 x W/4
                nn.BatchNorm2d(ndf * 2),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ndf * 2, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False), ## ndf * 4 x H/8 x W/8
                nn.BatchNorm2d(ndf * 2),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), ## ndf * 8 x H/16 x W/16
                nn.BatchNorm2d(ndf * 4),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ndf * 4, self.out_channels, kernel_size=4, stride=2, padding=1, bias=False),       ## ndf * 8 x H/32 x W/32
                # nn.BatchNorm2d(self.out_channels),
                # nn.LeakyReLU(0.2, inplace=True),
                # nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1, bias=False),       ## 1 x H/64 x W/64
                nn.Dropout2d(dropout)
            ]

        # if use_dropout:
        #     self.conv += [nn.Dropout2d(0.5)]
        self.conv = nn.Sequential(*self.conv)

        if self.minibatch_discrimination:
            self.mbd = MiniBatchDiscrimination(self.out_channels * 2 * 2, self.out_channels, 16, batch_size)
            # self.fc_last = nn.Linear(out_channels * 2 * 2 + out_channels // 2, 1)

    def forward(self, image, word_vectors):
        b, _, h, w = image.size()
        wv_out = self.fc(word_vectors)
        wv_out = wv_out.view(b, 1, h, w)
        stacked = torch.cat((image, wv_out), dim=1)
        print ("Dis S: 1", stacked.size())
        out = self.conv(stacked)
        print ("Dis S: 2", out.size())
        if self.minibatch_discrimination:
            out = out.view(-1, self.out_channels * 2 * 2)
            out = torch.cat((out, self.mbd(out)), dim=1)
            # return self.fc_last(out)
        return out

class DiscriminatorDecider(nn.Module):
    def __init__(self, config):
        super(DiscriminatorDecider, self).__init__()
        ndf = config.ND_DEC_F
        self.ndf = ndf
        n_channels = config.N_CHANNELS
        self.out_channels = config.OUT_CHANNELS
        use_spectral = config.USE_SPECTRAL_NORM
        dropout = config.D_DROPOUT
        self.minibatch_discrimination = config.MINIBATCH_DISCRIMINATION
        batch_size = config.BATCH_SIZE

        if use_spectral:
            self.conv = [                                                       ## 3 x H x W
                spectral_norm(nn.Conv2d(n_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False)),   ## ndf x H/2 x W/2
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)),      ## ndf * 2 x H/4 x W/4
                nn.BatchNorm2d(ndf * 2),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)),  ## ndf * 4 x H/8 x W/8
                nn.BatchNorm2d(ndf * 4),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)),  ## ndf * 4 x H/16 x W/16
                nn.BatchNorm2d(ndf * 4),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False)),  ## ndf * 8 x H/32 x W/32
                nn.BatchNorm2d(ndf * 8),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                spectral_norm(nn.Conv2d(ndf * 8, self.out_channels, kernel_size=4, stride=2, padding=1, bias=False)),       ## 1 x H/64 x W/64
                nn.Dropout2d(dropout)
            ]
        else: 
            self.conv = [                                                       ## 3 x H x W
                nn.Conv2d(n_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),           ## ndf x H/2 x W/2
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),     ## ndf * 2 x H/4 x W/4
                nn.BatchNorm2d(ndf * 2),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), ## ndf * 4 x H/8 x W/8
                nn.BatchNorm2d(ndf * 4),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False), ## ndf * 8 x H/16 x W/16
                nn.BatchNorm2d(ndf * 8),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),       ## ndf * 8 x H/32 x W/32
                nn.BatchNorm2d(ndf * 8),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ndf * 8, self.out_channels, kernel_size=4, stride=2, padding=1, bias=False),       ## 1 x H/64 x W/64
                nn.Dropout2d(dropout)
            ]
                
            self.conv1= nn.Conv2d(n_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False)           ## ndf x H/2 x W/2
#                nn.LeakyReLU(0.2, inplace=True),
            self.leaky1= nn.LeakyReLU(0.2, inplace=False)
            self.conv2= nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)     ## ndf * 2 x H/4 x W/4
            self.norm1= nn.BatchNorm2d(ndf * 2)
#                nn.LeakyReLU(0.2, inplace=True),
            self.leaky2= nn.LeakyReLU(0.2, inplace=False)
            self.conv3=  nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False) ## ndf * 4 x H/8 x W/8
            self.norm2=  nn.BatchNorm2d(ndf * 4)
#                nn.LeakyReLU(0.2, inplace=True),
            self.leaky3= nn.LeakyReLU(0.2, inplace=False)
            self.conv4= nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False) ## ndf * 8 x H/16 x W/16
            self.norm3= nn.BatchNorm2d(ndf * 8)
#                nn.LeakyReLU(0.2, inplace=True),
            self.leaky4= nn.LeakyReLU(0.2, inplace=False)
            self.conv5= nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False)       ## ndf * 8 x H/32 x W/32
            self.norm4= nn.BatchNorm2d(ndf * 8)
#                nn.LeakyReLU(0.2, inplace=True),
            self.leaky5= nn.LeakyReLU(0.2, inplace=False)
            self.conv6= nn.Conv2d(ndf * 8, self.out_channels, kernel_size=4, stride=2, padding=1, bias=False)       ## 1 x H/64 x W/64
#                nn.Dropout2d(dropout)



        self.conv = nn.Sequential(*self.conv)

        if self.minibatch_discrimination:
            self.mbd = MiniBatchDiscrimination(self.out_channels * 2 * 2, self.out_channels, self.ndf, batch_size)

    def forward(self, image):
        out= self.conv1(image)
        print ("Dis Dec 1: spec1", out.size())  
        out= self.leaky1(out)
        print ("Dis Dec 1: spec2", out.size())  
        out= self.conv2(out)
        print ("Dis Dec 1: spec3", out.size())  
        out= self.norm1(out)
        print ("Dis Dec 1: spec4", out.size())   
        out= self.leaky2(out)
        print ("Dis Dec 1: spec5", out.size())  
        out= self.conv3(out)
        print ("Dis Dec 1: spec6", out.size())  
        out= self.norm2(out)
        print ("Dis Dec 1: spec7", out.size())  
        out= self.leaky3(out)
        print ("Dis Dec 1: spec8", out.size())  
        out= self.conv4(out)
        print ("Dis Dec 1: spec9", out.size())  
        out= self.norm3(out)
        print ("Dis Dec 1: spec10", out.size())  
        out= self.leaky4(out)
        print ("Dis Dec 1: spec11", out.size())  
        out= self.conv5(out)
        print ("Dis Dec 1: spec12", out.size())  
        out= self.norm4(out)
        print ("Dis Dec 1: spec13", out.size())  
        out= self.leaky5(out)
        print ("Dis Dec 1: spec14", out.size())  
        out= self.conv5(out)
        print ("Dis Dec 1: spec15", out.size())       
#        out = self.conv(image).clone()
#        print ("Dis Dec 1: 1", out.size())   
        if self.minibatch_discrimination:
            out = out.view(-1, self.out_channels * 2 * 2)
            print ("Dis Dec 1: 2", out.size()) 
            out = torch.cat((out, self.mbd(out)), dim=1)
            print ("Dis Dec 1: 3", out.size()) 
        print ("Dis Dec 1 exit")   
        return out

class DiscriminatorDecider2(nn.Module):
    def __init__(self, config):
        super(DiscriminatorDecider2, self).__init__()
        ndf = config.ND_DEC_F
        self.ndf = ndf
        n_channels = config.N_CHANNELS
        self.out_channels = config.OUT_CHANNELS
        use_spectral = config.USE_SPECTRAL_NORM
        dropout = config.D_DROPOUT
        self.minibatch_discrimination = config.MINIBATCH_DISCRIMINATION
        batch_size = config.BATCH_SIZE

        if use_spectral:
            self.conv = [                                                       ## 3 x H x W
                spectral_norm(nn.Conv2d(n_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False)),   ## ndf x H/2 x W/2
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)),      ## ndf * 2 x H/4 x W/4
                nn.BatchNorm2d(ndf * 2),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                spectral_norm(nn.Conv2d(ndf * 2, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)),  ## ndf * 4 x H/8 x W/8
                nn.BatchNorm2d(ndf * 2),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)),  ## ndf * 4 x H/16 x W/16
                nn.BatchNorm2d(ndf * 4),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)),  ## ndf * 8 x H/32 x W/32
                nn.BatchNorm2d(ndf * 4),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False)),                 ## ndf * 8 x H/64 x W/64
                nn.BatchNorm2d(ndf * 8),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                spectral_norm(nn.Conv2d(ndf * 8, self.out_channels, kernel_size=4, stride=2, padding=1, bias=False)),       ## 1 x H/128 x W/128
                nn.Dropout2d(dropout)
            ]
        else:
            self.conv = [                                                       ## 3 x H x W
                nn.Conv2d(n_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),           ## ndf x H/2 x W/2
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),     ## ndf * 2 x H/4 x W/4
                nn.BatchNorm2d(ndf * 2),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ndf * 2, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False), ## ndf * 4 x H/8 x W/8
                nn.BatchNorm2d(ndf * 2),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), ## ndf * 8 x H/16 x W/16
                nn.BatchNorm2d(ndf * 4),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ndf * 4, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),       ## ndf * 8 x H/32 x W/32
                nn.BatchNorm2d(ndf * 4),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),       ## ndf * 8 x H/64 x W/64
                nn.BatchNorm2d(ndf * 8),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ndf * 8, self.out_channels, kernel_size=4, stride=2, padding=1, bias=False),       ## 1 x H/128 x W/128
                nn.Dropout2d(dropout)
            ]

        self.conv = nn.Sequential(*self.conv)

        if self.minibatch_discrimination:
            self.mbd = MiniBatchDiscrimination(self.out_channels * 2 * 2, self.out_channels, self.ndf, batch_size)

    def forward(self, image):
        out = self.conv(image)
        print ("Dis Dec 2: 1", out.size()) 
        if self.minibatch_discrimination:
            out = out.view(-1, self.out_channels * 2 * 2)
            print ("Dis Dec 2: 2", out.size()) 
            out = torch.cat((out, self.mbd(out)), dim=1)
            print ("Dis Dec 2: 3", out.size()) 
        return out

class MiniBatchDiscrimination(nn.Module):
    def __init__(self, A, B, C, batch_size):
        super(MiniBatchDiscrimination, self).__init__()
        self.feat_num = A
        self.out_size = B
        self.row_size = C
        self.N = batch_size
        self.T = torch.nn.parameter.Parameter(torch.Tensor(A, B, C))
        self.reset_parameters()

    def forward(self, x):
        Ms = x.mm(self.T.view(self.feat_num, self.out_size * self.row_size))
        print ("Mini D: 1", Ms.size()) 
        Ms = Ms.view(-1, self.out_size, self.row_size)
        print ("Mini D: 2", Ms.size()) 

        out_tensor = []
        for i in range(Ms.size()[0]):

            out_i = None
            for j in range(Ms.size()[0]):
                o_i = torch.sum(torch.abs(Ms[i, :, :] - Ms[j, :, :]), 1)
                o_i = torch.exp(-o_i)
                if out_i is None:
                    out_i = o_i
                else:
                    out_i = out_i + o_i

            out_tensor.append(out_i)

        out_T = torch.cat(tuple(out_tensor)).view(Ms.size()[0], self.out_size)
        print ("Mini D: 4", out_T.size()) 
        return out_T

    def reset_parameters(self):
        stddev = 1 / self.feat_num
        self.T.data.uniform_(stddev)

class GeneratorRefiner(nn.Module):

    def __init__(self, config):
        super(GeneratorRefiner, self).__init__()

        n_channels = config.N_CHANNELS
        ngf = config.NG_REF_F
        self.ngf = ngf
        self.norm_layer = config.NORM_LAYER
        self.use_spectral = config.USE_SPECTRAL_NORM
        self.dropout = config.G_DROPOUT
        self.padding_type = config.PADDING_TYPE
        n_blocks = config.N_BLOCKS
        assert(n_blocks >= 0)

        if type(self.norm_layer) == functools.partial:
            self.use_bias = self.norm_layer.func == nn.InstanceNorm2d
        else:
            self.use_bias = self.norm_layer == nn.InstanceNorm2d

        if self.use_spectral:
            self.downsample = [                                                         ## 3 x H x W
                spectral_norm(nn.Conv2d(n_channels, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False)),      ## ngf x H/2 x W/2
                nn.BatchNorm2d(ngf * 2),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                spectral_norm(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False)),       ## ngf x H/4 x W/4
                nn.BatchNorm2d(ngf * 4),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                spectral_norm(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False)),            ## ngf x H/8 x W/8
                # nn.BatchNorm2d(ngf * 4),
                # nn.LeakyReLU(0.2, inplace=True),
                # nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),                 ## ngf x H/16 x W/16
                # nn.BatchNorm2d(ngf * 8),
                # nn.LeakyReLU(0.2, inplace=True),
                # nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),                 ## ngf x H/32 x W/32
                nn.Dropout2d(self.dropout)
            ]
        else:
            self.downsample = [                                                         ## 3 x H x W
                nn.Conv2d(n_channels, ngf, kernel_size=4, stride=2, padding=1, bias=False),      ## ngf x H/2 x W/2
                nn.BatchNorm2d(ngf),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),       ## ngf x H/4 x W/4
                nn.BatchNorm2d(ngf * 2),
#                nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),            ## ngf x H/8 x W/8
                # nn.BatchNorm2d(ngf * 4),
                # nn.LeakyReLU(0.2, inplace=True),
                # nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),                 ## ngf x H/16 x W/16
                # nn.BatchNorm2d(ngf * 8),
                # nn.LeakyReLU(0.2, inplace=True),
                # nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),                 ## ngf x H/32 x W/32
                nn.Dropout2d(self.dropout)
            ]

        # if self.dropout:
        #     self.downsample += [nn.Dropout2d(0.5)]
        self.downsample = nn.Sequential(*self.downsample)

        # self.block1 = self._build_blocks(ngf * 4, n_blocks)
        self.upsample1 = upsample_block(ngf * 8, ngf * 4, dropout=self.dropout)           # ngf x H x W -> ngf/2 x 2H x 2W
        self.upsample2 = upsample_block(ngf * 4, ngf * 2, dropout=self.dropout)      # -> ngf/4 x 4H x 4W
        self.upsample3 = upsample_block(ngf * 2, ngf, dropout=self.dropout)      # -> ngf/8 x 8H x 8W
        # self.upsample4 = upsample_block(ngf, ngf // 2, dropout=self.dropout)     # -> ngf/16 x 16H x 16W
        # self.upsample5 = upsample_block(ngf // 2, 3, dropout=self.dropout)     # -> ngf/16 x 32H x 32W
        self.block = self._build_blocks(ngf, n_blocks)

        if self.use_spectral:
            self.finish = nn.Sequential(
                                       nn.ReflectionPad2d(3),
                                       spectral_norm(nn.Conv2d(ngf, 3, kernel_size=7, padding=0)),    # 3 x 16H x 16W
                                       nn.Tanh()
                                       )
        else:
            self.finish = nn.Sequential(
                                       nn.ReflectionPad2d(3),
                                       nn.Conv2d(ngf // 2, 3, kernel_size=7, padding=0),    # 3 x 16H x 16W
                                       nn.Tanh()
                                       )

    def _build_blocks(self, channel, n_blocks):
        blocks = []
        for i in range(n_blocks):       # add ResNet blocks
#            blocks += [ResnetBlock(channel,
            blocks = blocks + [ResnetBlock(channel,
                                    padding_type=self.padding_type,
                                    norm_layer=self.norm_layer,
                                    dropout=self.dropout,
                                    use_bias=self.use_bias,
                                    use_spectral=self.use_spectral)
                                    ]
        blocks = nn.Sequential(*blocks)                  # -> ngf x H x W
        return blocks

    def forward(self, image):
        print ("G Ref: 1", image.size())
        out = self.downsample(image)
        print ("G Ref: 2", out.size())
        out = self.upsample1(out)
        print ("G Ref: 3", out.size())
        out = self.upsample2(out)
        print ("G Ref: 4", out.size())
        out = self.upsample3(out)
        print ("G Ref: 5", out.size())
        out = self.block(out)
        print ("G Ref: 6", out.size())
        out = self.finish(out)
        print ("G Ref: 7", out.size())

        return out

class GeneratorSimple(nn.Module):
    def __init__(self, config):
        super(GeneratorSimple, self).__init__()
        
        n_input = config.N_INPUT
        ngf = config.NGF

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(n_input, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),       ## ngf * 8 x 4 x 4 
            nn.BatchNorm2d(ngf * 8),
#            nn.ReLU(True),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),      ## ngf * 4 x 8 x 8
            nn.BatchNorm2d(ngf * 4),
#            nn.ReLU(True),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),  ## ngf * 2 x 16 x 16
            nn.BatchNorm2d(ngf * 2),
#            nn.ReLU(True),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),        ## 3 x 32 x 32
            nn.BatchNorm2d(ngf),
#            nn.ReLU(True),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),     ## 3 x 64 x 64
            nn.Tanh()
        )

    def forward(self, x):
        print ("G Simple: 1", self.conv(x).size()) 
        return self.conv(x)

class DiscriminatorSimple(nn.Module):
    def __init__(self, config):
        super(DiscriminatorSimple, self).__init__()

        n_input = config.N_INPUT
        ndf = config.NDF

        self.conv = nn.Sequential(                                                       ## 3 x H x W
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False),           ## ndf x H/2 x W/2
#            nn.LeakyReLU(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),     ## ndf * 2 x H/4 x W/4
            nn.BatchNorm2d(ndf * 2),
#            nn.LeakyReLU(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False), ## ndf * 4 x H/8 x W/8
            nn.BatchNorm2d(ndf * 4),
#            nn.LeakyReLU(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False), ## ndf * 8 x H/16 x W/16
            nn.BatchNorm2d(ndf * 8),
#            nn.LeakyReLU(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1, bias=False),       ## 1 x H/32 x W/32
        )

    def forward(self, x):
        print ("D Simple: 1", self.conv(x).size()) 
        return self.conv(x)

class GeneratorStack(nn.Module):

    def __init__(self, config):
        super(GeneratorStack, self).__init__()

        n_input = config.N_INPUT
        ngf = config.NGF * 8
        n_channel = config.N_CHANNELS
        self.n_input = n_input
        self.ngf = ngf

        self.fc = nn.Sequential(
                               nn.Linear(n_input, ngf * 4 * 4, bias=False),
                               nn.BatchNorm1d(ngf * 4 * 4),
                            #    nn.InstanceNorm1d(ngf * 4 * 4),
#                               nn.ReLU(True)
                                nn.ReLU(inplace=False)
                               )                                 # -> ngf x H x W

        self.upsample1 = upsample_block(ngf, ngf // 2)           # ngf x H x W -> ngf/2 x 2H x 2W
        self.upsample2 = upsample_block(ngf // 2, ngf // 4)      # -> ngf/4 x 4H x 4W
        self.upsample3 = upsample_block(ngf // 4, ngf // 8)      # -> ngf/8 x 8H x 8W
        self.upsample4 = upsample_block(ngf // 8, ngf // 16)     # -> ngf/16 x 16H x 16W
        self.upsample5 = upsample_block(ngf // 16, ngf // 32)     # -> ngf/32 x 32H x 32W

        self.dropout = nn.Dropout2d(0.5)

        self.image = nn.Sequential(
                                  conv3x3(ngf // 32, n_channel),
                                  nn.Tanh()
                                  )                              # -> 3 x 16H x 16W

    def forward(self, word_vectors):
        # out = self.dropout(word_vectors)
        out = self.fc(word_vectors)
        print ("G Stack: 1", out.size()) 
        out = out.view(-1, self.ngf, 4, 4)
        print ("G Stack: 2", out.size()) 
        out = self.upsample1(out)
        print ("G Stack: 3", out.size()) 
        out = self.upsample2(out)
        print ("G Stack: 4", out.size()) 
        out = self.upsample3(out)
        print ("G Stack: 5", out.size()) 
        out = self.upsample4(out)
        print ("G Stack: 6", out.size()) 
        out = self.upsample5(out)
        print ("G Stack: 7", out.size()) 
        out = self.dropout(out)
        print ("G Stack: 8", out.size()) 
        out = self.image(out)
        print ("G Stack: 9", out.size()) 

        return out

class GeneratorUNet(nn.Module):
    def __init__(self, config):
        super(GeneratorUNet, self).__init__()

        n_channels = config.N_CHANNELS
        n_input = config.N_INPUT
        ngf = config.NGF
        use_spectral = config.USE_SPECTRAL_NORM
        dropout = config.G_DROPOUT

        self.fc = nn.Sequential(
                               nn.Linear(n_input, ngf * 4 * 4, bias=False),
#                               nn.ReLU(True),
                                nn.ReLU(inplace=False),
                               nn.Dropout(dropout)
                               )

        self.up1 = UNetUp(ngf, ngf)
        self.up2 = UNetUp(ngf, ngf // 2)
        self.up3 = UNetUp(ngf // 2, ngf // 4)
        self.up4 = UNetUp(ngf // 4, ngf // 8, dropout=dropout)

        # self.down1 = UNetDown(ngf // 8, ngf // 4, dropout=dropout, use_spectral=use_spectral)
        # self.down2 = UNetDown(ngf // 4 + ngf // 4, ngf // 4, dropout=dropout, use_spectral=use_spectral)
        # self.down3 = UNetDown(ngf // 4 + ngf // 2, ngf // 4, dropout=dropout, use_spectral=use_spectral)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(ngf // 4 + ngf, n_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        u1 = self.up1(x)
        print ("G Unet: 1", u1.size()) 
        u2 = self.up2(u1)
        print ("G Unet: 2", u2.size()) 
        u3 = self.up3(u2)
        print ("G Unet: 3", u3.size()) 
        u4 = self.up4(u3)
        print ("G Unet: 4", u4.size()) 
        # d1 = self.down1(u4, u3)
        # d2 = self.down2(d1, u2)
        # d3 = self.down3(d2, u1)

        return self.final(u4)

if __name__ == '__main__':
    from config import Config
    config = Config()
    G = GeneratorResNet(config)
    image = torch.Tensor(2, 4096)
    G(image)