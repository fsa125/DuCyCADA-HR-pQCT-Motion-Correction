
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from CinCGAN_pytorch.utils import *
import torch.nn.functional as F
class ResnetGenerator(nn.Module):
    #Generator architecture
    def __init__(self, input_nc=1, output_nc=1, inter_nc=64, n_blocks=6, img_size = 256, use_bias = False, rs_norm = 'BN', padding_type = 'zero', dsple = False, scale_factor=1):
        # input_nc(int) -- The number of channels of input img
        # output_nc(int) -- The number of channels of output img
        # inter_nc(int) -- The number of filters of intermediate layers
        # n_blocks(int) -- The number of resnet blocks
        # img_size(int) -- Input image size
        # use_bias(bool) -- Whether to use bias on conv layer or not
        # rs_norm(str) -- The type of normalization method of ResnetBlock. BN : Batch Normalization, IN : Instance Normalization, else : none
        # padding_type(str) -- The name of padding layer: reflect | replicate | zero
        # dsple(bool) -- Whether to downsample or maintain input image. Set it true for G3.
        # scale_factor(int) -- Scale factor, 2 / 4
        super(ResnetGenerator, self).__init__()
        
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.inter_nc = inter_nc
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.use_bias = use_bias
        self.rs_norm = rs_norm
        self.padding_type = padding_type
        self.dsple = dsple
        self.scale_factor = scale_factor
        
        # Input blocks
        InBlock = []
        
        InBlock += [nn.Conv2d(input_nc, inter_nc, kernel_size=7, stride=1, padding=3, bias=self.use_bias),
                     nn.LeakyReLU(0.2)]
        InBlock += [nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=2 if self.dsple and self.scale_factor==4 else 1, padding=1, bias=self.use_bias),
                     nn.LeakyReLU(0.2)] #changed
        InBlock += [nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=2 if self.dsple else 1, padding=1, bias=self.use_bias),
                     nn.LeakyReLU(0.2)]
        
        # ResnetBlocks
        ResnetBlocks = []
        
        for i in range(n_blocks):
            ResnetBlocks += [ResnetBlock(inter_nc, self.padding_type, self.rs_norm, self.use_bias)]
        
        # Output block
        OutBlock = []
        
        OutBlock += [nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias),
                     nn.LeakyReLU(0.2)]
        OutBlock += [nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias),
                     nn.LeakyReLU(0.2)]
        OutBlock += [nn.Conv2d(inter_nc, output_nc, kernel_size=7, stride=1, padding=3, bias=self.use_bias),
                     nn.LeakyReLU(0.2)]
        
        self.InBlock = nn.Sequential(*InBlock)
        self.ResnetBlocks = nn.Sequential(*ResnetBlocks)
        self.OutBlock = nn.Sequential(*OutBlock)
    def forward(self,x):
        #print(x.shape)
        out = self.InBlock(x)
        out = self.ResnetBlocks(out)
        out = self.OutBlock(out)
        
        return out
    

class ResnetGenerator_VAE(nn.Module):
    #Generator architecture
    def __init__(self, input_nc=1, output_nc=1, inter_nc=64, n_blocks=6, img_size = 256, use_bias = False, rs_norm = 'BN', padding_type = 'zero', dsple = True, scale_factor=1, latent_dim = 256):
        # input_nc(int) -- The number of channels of input img
        # output_nc(int) -- The number of channels of output img
        # inter_nc(int) -- The number of filters of intermediate layers
        # n_blocks(int) -- The number of resnet blocks
        # img_size(int) -- Input image size
        # use_bias(bool) -- Whether to use bias on conv layer or not
        # rs_norm(str) -- The type of normalization method of ResnetBlock. BN : Batch Normalization, IN : Instance Normalization, else : none
        # padding_type(str) -- The name of padding layer: reflect | replicate | zero
        # dsple(bool) -- Whether to downsample or maintain input image. Set it true for G3.
        # scale_factor(int) -- Scale factor, 2 / 4
        super(ResnetGenerator_VAE, self).__init__()
        self.latent_dim = latent_dim #Add encoder logic that outputs mu and logvar

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.inter_nc = inter_nc
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.use_bias = use_bias
        self.rs_norm = rs_norm
        self.padding_type = padding_type
        self.dsple = dsple
        self.scale_factor = scale_factor
        self.spatial_dim = img_size // 4 if dsple else img_size
        self.feature_size = inter_nc 
        
        # Input blocks
        InBlock = []
        
        InBlock += [nn.Conv2d(input_nc, inter_nc, kernel_size=7, stride=1, padding=3, bias=self.use_bias),
                     nn.LeakyReLU(0.2)]
        InBlock += [nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=2 if self.dsple and self.scale_factor==1 else 1, padding=1, bias=self.use_bias),
                     nn.LeakyReLU(0.2)] #changed
        InBlock += [nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=2 if self.dsple else 1, padding=1, bias=self.use_bias),
                     nn.LeakyReLU(0.2)]
        
        # ResnetBlocks
        ResnetBlocks = []
        
        for i in range(n_blocks):
            ResnetBlocks += [ResnetBlock(inter_nc, self.padding_type, self.rs_norm, self.use_bias)]


         # μ and logσ² from 64×64×64
        flattened_dim = inter_nc * self.spatial_dim * self.spatial_dim
        self.fc_mu = nn.Linear(flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim, latent_dim)

        # Project back from latent vector
        self.fc_decode = nn.Linear(latent_dim, flattened_dim)
        
        # Output block
        #OutBlock = []
        
        #OutBlock += [nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias),
         #            nn.LeakyReLU(0.2)]
        #OutBlock += [nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias),
          #           nn.LeakyReLU(0.2)]
        #OutBlock += [nn.Conv2d(inter_nc, output_nc, kernel_size=7, stride=1, padding=3, bias=self.use_bias),
         #            nn.LeakyReLU(0.2)]
        
        OutBlock = []

#       First upsampling + conv + activation
        OutBlock += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
             nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias),
             nn.LeakyReLU(0.2)]

        # Second upsampling + conv + activation
        OutBlock += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
             nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias),
             nn.LeakyReLU(0.2)]

        # Final convolution to reduce channels
        OutBlock += [nn.Conv2d(inter_nc, output_nc, kernel_size=7, stride=1, padding=3, bias=self.use_bias),
             nn.Tanh()]  # or LeakyReLU/Sigmoid depending on your use case
        
        self.InBlock = nn.Sequential(*InBlock)
        self.ResnetBlocks = nn.Sequential(*ResnetBlocks)
        self.OutBlock = nn.Sequential(*OutBlock)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self,x):
        #print(x.shape)
        out = self.InBlock(x)
        out = self.ResnetBlocks(out)

        batch_size = out.size(0)
        out_flat = out.view(batch_size, -1)

        mu = self.fc_mu(out_flat)
        logvar = self.fc_logvar(out_flat)
        z = self.reparameterize(mu, logvar)

        out_decode = self.fc_decode(z).view(batch_size, self.feature_size, self.spatial_dim, self.spatial_dim)
        out_img = self.OutBlock(out_decode)

        return out_img, mu, logvar, z
    
class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512):
        super(LatentDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)
    
  
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_type, use_bias):
        # dim(int) -- The number of channels in the resnet blocks
        # padding_type(str) -- The name of padding layer: reflect | replicate | zero
        # norm_type(str) -- The type of normalization method. BN : Batch Normalization, IN : Instance Normalization, else : none
        # use_bias -- Whether to use bias on conv layer or not
        super(ResnetBlock, self).__init__()
        
        conv_block = []
        
        # Padding
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            
        if norm_type=='BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type=='IN':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError('Normalization [%s] is not implemented' % norm_type)
        
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                       norm_layer(dim), 
                       nn.LeakyReLU(0.2)]

        
        # Padding
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                       norm_layer(dim), 
                       nn.LeakyReLU(0.2)]
        
        self.conv_block = nn.Sequential(*conv_block)
        
    def forward(self, x):
        out = self.conv_block(x)
        
        # Skip connection
        out = out + x
        
        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc=1, norm_type = 'BN', use_bias = True, is_inner=True, scale_factor=4):
        # input_nc(int) -- The number of channels of input img
        # norm_type(str) -- The type of normalization method. BN : Batch Normalization, IN : Instance Normalization, else : none
        # use_bias(bool) -- Whether to use bias or not
        # is_inner(bool) -- True : For inner cycle, False : For outer cycle
        # scale_factor(int) -- Scale factor, 2 / 4

        super(Discriminator, self).__init__()
        
        if norm_type=='BN':
            norm_layer = nn.BatchNorm2d
            use_bias = False # There is no need to use bias because BN already has shift parameter.
        elif norm_type=='IN':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError('Normalization [%s] is not implemented' % norm_type)
        
        if is_inner == True:
            s = 1
        elif is_inner == False:
            s = 2
        else:
            raise NotImplementedError('is_inner must be boolean.')
        
        nfil_mul = 64
        p=0 # Why 1???
        layers = []
        layers += [nn.Conv2d(input_nc, nfil_mul, kernel_size=4, stride = 2 if is_inner==True and scale_factor==2 else s, padding=p, bias=use_bias), 
                       nn.LeakyReLU(0.2)] # changed
        layers += [nn.Conv2d(nfil_mul, nfil_mul*2, kernel_size=4, stride = s, padding=p, bias=use_bias), 
                       norm_layer(nfil_mul*2), 
                       nn.LeakyReLU(0.2)]
        layers += [nn.Conv2d(nfil_mul*2, nfil_mul*4, kernel_size=4, stride = s, padding=p, bias=use_bias), 
                       norm_layer(nfil_mul*4), 
                       nn.LeakyReLU(0.2)]
        layers += [nn.Conv2d(nfil_mul*4, nfil_mul*8, kernel_size=4, stride = 1, padding=p, bias=use_bias), 
                       norm_layer(nfil_mul*8), 
                       nn.LeakyReLU(0.2)]
        layers += [nn.Conv2d(nfil_mul*8, 1, kernel_size=4, stride = 1, padding=p, bias=use_bias), 
                       nn.LeakyReLU(0.2)]
        self.layers = nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.layers(x)
        
        return out # Predicted values of each patches
    

class DiscriminatorDA(nn.Module):
    def __init__(self, input_nc=1, norm_type = 'BN', use_bias = True, is_inner=True, scale_factor=4):
        # input_nc(int) -- The number of channels of input img
        # norm_type(str) -- The type of normalization method. BN : Batch Normalization, IN : Instance Normalization, else : none
        # use_bias(bool) -- Whether to use bias or not
        # is_inner(bool) -- True : For inner cycle, False : For outer cycle
        # scale_factor(int) -- Scale factor, 2 / 4

        super(DiscriminatorDA, self).__init__()
        
        if norm_type=='BN':
            norm_layer = nn.BatchNorm2d
            use_bias = False # There is no need to use bias because BN already has shift parameter.
        elif norm_type=='IN':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError('Normalization [%s] is not implemented' % norm_type)
        
        if is_inner == True:
            s = 1
        elif is_inner == False:
            s = 2
        else:
            raise NotImplementedError('is_inner must be boolean.')
        
        nfil_mul = 64
        p=0 # Why 1???
        layers = []
        layers += [nn.Conv2d(input_nc, nfil_mul, kernel_size=4, stride = 2 if is_inner==True and scale_factor==2 else s, padding=p, bias=use_bias), 
                       nn.LeakyReLU(0.2)] # changed
        layers += [nn.Conv2d(nfil_mul, nfil_mul*2, kernel_size=4, stride = s, padding=p, bias=use_bias), 
                       norm_layer(nfil_mul*2), 
                       nn.LeakyReLU(0.2)]
        layers += [nn.Conv2d(nfil_mul*2, nfil_mul*4, kernel_size=4, stride = s, padding=p, bias=use_bias), 
                       norm_layer(nfil_mul*4), 
                       nn.LeakyReLU(0.2)]
        layers += [nn.Conv2d(nfil_mul*4, nfil_mul*8, kernel_size=4, stride = 1, padding=p, bias=use_bias), 
                       norm_layer(nfil_mul*8), 
                       nn.LeakyReLU(0.2)]
        layers += [nn.Conv2d(nfil_mul*8, 2, kernel_size=4, stride = 1, padding=p, bias=use_bias), 
                       nn.LeakyReLU(0.2)]
        self.layers = nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.layers(x)
        
        return out # Predicted values of each patches

        


class UNetDiscriminator(nn.Module):
    def __init__(self, in_channels=1, base_filters=64):
        super(UNetDiscriminator, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, base_filters)
        self.enc2 = self.conv_block(base_filters, base_filters * 2)
        self.enc3 = self.conv_block(base_filters * 2, base_filters * 4)
        self.enc4 = self.conv_block(base_filters * 4, base_filters * 4)

        # Bottleneck
        self.bottleneck = self.conv_block(base_filters * 4, base_filters * 8)

        # Decoder with skip connections
        self.dec4 = self.up_block(base_filters * 8, base_filters * 4)
        self.dec3 = self.up_block(base_filters * 8, base_filters * 4)  # because of skip connection
        self.dec2 = self.up_block(base_filters * 8, base_filters*2)
        self.dec1 = nn.Conv2d(base_filters * 4, 1, kernel_size=1)  # final layer to get 1 channel output

    def conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)  # 128x128 #channels 64
        e2 = self.enc2(e1) # 64x64 #channels 128
        e3 = self.enc3(e2) # 32x32 #channels 256
        e4 = self.enc4(e3) # 16x16 #channels 256

        # Bottleneck
        b = self.bottleneck(e4)  # 8x8X512

        # Decoding path with skip connections
        d4 = self.dec4(b)         # 16x16X256
        d4 = torch.cat([d4, e4], dim=1) #channels 512

        d3 = self.dec3(d4) #channels 256        # 32x32
        d3 = torch.cat([d3, e3], dim=1) #512

        d2 = self.dec2(d3)        # 64x64X128
        d2 = torch.cat([d2, e2], dim=1)#128

        out = self.dec1(d2)       # 64x64 → output prediction map (PatchGAN style)

        return out
    

from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm



class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out
def test():
    from torchvision import transforms
    import matplotlib.pyplot as plt

    #device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load image
    img = pil_loader("/mnt/nas/data/track1/Corrupted-te-x/0917.png")
    
    # Transform image to torch Tensor. Normalized to 0~1 automatically.
    img = transforms.Resize((128,128))(img)
    img = transforms.ToTensor()(img).to(device)
    
    #print("Input shape : ",img.shape)

    # Feed to generator
    img = torch.unsqueeze(img,0)
    G1 = ResnetGenerator(dsple=True).to(device)
    fakeimgs = G1(img)

    #print("Fake image shape : ", fakeimgs.shape)

    # Feed to discriminator
    D1 = Discriminator(is_inner=True).to(device)
    out = D1(fakeimgs)
    #print(out.shape)



