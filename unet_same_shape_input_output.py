# import libraries
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as TF



# Define two convolutional network of size 3 * 3 (padded convolutions - though it is unpadded in the original architecture)
# The padding is so that we have an output of the same size after each convolutional network filter is applied on the input image 
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) 
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        print(x.shape)
        x = self.relu(self.conv2(x))
        print(x.shape)
        return x



# Define encoder
class Encoder(nn.Module):
    def __init__(self, in_channels, outchannels) -> None:
        super(Encoder, self).__init__()
        self.downsampling = ConvBlock(in_channels, outchannels)
        self.maxPooling = nn.MaxPool2d(kernel_size=2, stride=2, )

    def forward(self, x):
        print('Encoder in Action!!!')
        # perform downsampling
        skip_connections = self.downsampling(x)
        max_pool_feat = self.maxPooling(skip_connections)
        return skip_connections, max_pool_feat
    


# define decoder
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Decoder, self).__init__()
        self.upsampling = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, )
        self.downsampling = ConvBlock(in_channels, out_channels)


    def forward(self, x, skip_connections):
        print('Decoder in Action!!!')
        # perform upsampling
        x = self.upsampling(x)

        # if shape doesn't align, reshape x to skip_connections shape before concatenating them - this is not used in the Unet architecture
        # Note: we can't use copy and crop here because skip_connections is bigger than x and we are trying to reshape x into skip_connections shape so that input shape equals output shape
        x_shape = x.shape[-2:]
        skip_connect_shape = skip_connections.shape[-2:]

        if list(x_shape) != list(skip_connect_shape):
            # get the skip connection shape before cropping
            print(f'x_shape before cropping: {x.shape}, skip_connection shape : {skip_connections.shape}')

            # resize
            x = TF.resize(x, list(skip_connections.shape[-2:]))


        print(f'x_shape after cropping: {x.shape}')
        x =  torch.cat((x, skip_connections), dim=1)
        downsampled_feat = self.downsampling(x)
        return downsampled_feat


class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1) -> None:
        super(Unet, self).__init__()

        self.downsample_1 = Encoder(in_channels, 64)
        self.downsample_2 = Encoder(64, 128)
        self.downsample_3 = Encoder(128, 256)
        self.downsample_4 = Encoder(256, 512)
    
        self.bottleneck = ConvBlock(512, 1024)

        self.upsample_1 = Decoder(1024, 512)
        self.upsample_2 = Decoder(512, 256)
        self.upsample_3 = Decoder(256, 128)
        self.upsample_4 = Decoder(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):

        # encoder
        encoder_1, pool_1 = self.downsample_1(x)
        encoder_2, pool_2 = self.downsample_2(pool_1)
        encoder_3, pool_3 = self.downsample_3(pool_2)
        encoder_4, pool_4 = self.downsample_4(pool_3)

        # bottleneck
        bottleneck = self.bottleneck(pool_4)


        # decoder
        decoder_1 = self.upsample_1(bottleneck, encoder_4)
        decoder_2 = self.upsample_2(decoder_1, encoder_3)
        decoder_3 = self.upsample_3(decoder_2, encoder_2)
        decoder_4 = self.upsample_4(decoder_3, encoder_1)

        # final_conv
        output = self.final_conv(decoder_4)
        return output


########### testing the code
def test():
    # define input image
    input_image = torch.randn(1, 3, 572, 572)
    model = Unet(in_channels=3, out_channels=2)
    output = model(input_image)
    print(f'output shape: {output.shape}')
    assert output.shape == (1, 2, 572, 572), "ouput shape is different from input shape"


if __name__ == '__main__':
    test()