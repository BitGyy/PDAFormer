from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from thop import profile

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer

class MHCResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        groups: int = 1,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
            groups=groups,
        )
        
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        
        

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        
        out += residual
        out = self.lrelu(out)
        return out

class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        groups: int = 1,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
            groups=groups,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True,groups=groups,
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True
        if self.downsample:
            self.conv3 = get_conv_layer(
                spatial_dims, in_channels, out_channels, kernel_size=1, stride=stride, dropout=dropout, conv_only=True
            )
            self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if hasattr(self, "conv3"):
            residual = self.conv3(residual)
        if hasattr(self, "norm3"):
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class UnetBasicBlock(nn.Module):
    """
    A CNN module module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out


class UnetUpBlock(nn.Module):
    """
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        trans_bias: transposed convolution bias.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            conv_only=True,
            is_transposed=True,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out

#original UnetOut
# class UnetOutBlock(nn.Module):
#     def __init__(
#         self, spatial_dims: int, in_channels: int, out_channels: int, dropout: Optional[Union[Tuple, str, float]] = None
#     ):
#         super().__init__()
#         self.conv = get_conv_layer(
#             spatial_dims, in_channels, out_channels, kernel_size=1, stride=1, dropout=dropout, bias=True, conv_only=True
#         )
        

#     def forward(self, inp):
#         return self.conv(inp)
    
class UnetOutBlock(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, dropout: Optional[Union[Tuple, str, float]] = None
    ):
        super().__init__()
        self.conv1 = get_conv_layer(spatial_dims, in_channels, in_channels, kernel_size=3, stride=1)
        self.conv2 = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=1, stride=1, dropout=dropout, bias=True, conv_only=True)
        

    def forward(self, inp):
        return self.conv2(self.conv1(inp))


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Union[Tuple, str] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
    groups: int = 1,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        groups=groups,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:

    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], padding: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]

#res conv attn
class ConvAttn(nn.Module):
    def __init__(
            self,
            ch,
            ch_out,
    ) -> None:
        super().__init__()
        self.c1 = nn.Conv3d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv3d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv3d(in_channels=ch, out_channels=ch_out, kernel_size=1, stride=1, padding=0)
        self.norm = nn.InstanceNorm3d(ch)
        self.norm2 = nn.InstanceNorm3d(ch_out)
        self.activate = nn.LeakyReLU(negative_slope=5e-2)

        
    def forward(self, x):
        x_1 = self.activate(self.norm(self.c1(x)))
        x_2 = self.norm(self.c2(x_1))
        output = self.activate(x + x_2)
#         output = self.activate(x_2)
        output = self.norm2(self.c3(output))
        return output

# class ConvAttn(nn.Module):
#     def __init__(
#             self,
#             ch,
#             ch_out,
#     ) -> None:
#         super().__init__()
#         self.c1 = nn.Conv3d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1)
#         self.c2 = nn.Conv3d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1)
#         self.c3 = nn.Conv3d(in_channels=ch, out_channels=ch_out, kernel_size=1, stride=1, padding=0)
#         self.norm = nn.InstanceNorm3d(ch)
#         self.norm2 = nn.InstanceNorm3d(ch_out)
#         self.activate = nn.LeakyReLU(negative_slope=5e-2)
        
#     def forward(self, x):
#         x_1 = self.activate(self.norm(self.c1(x)))
#         x_2 = self.norm(self.c2(x_1))
#         output = self.activate(x_2)
#         output = self.norm2(self.c3(output))
#         return output


class mp_ap(nn.Module):
    def __init__(
        self,
        i : int,
    ) -> None:
        super().__init__()
        self.maxpool = nn.MaxPool3d(kernel_size = 2**(i), stride = 2**(i))
        self.avgpool = nn.AvgPool3d(kernel_size = 2**(i), stride = 2**(i))
    def forward(self, x):
        output = self.maxpool(x) + self.avgpool(x)
        return output

class Pool(nn.Module):
    def __init__(
        self,
        i: int,
        ch: int,
        k: int,
    ) -> None:
        super().__init__()
        self.i = i
        self.ch = ch
        self.pool_layers = nn.ModuleList()
        for num in range(i+1):
            self.pool_layers.append(mp_ap(num)) 
    def forward(self, x):
        enc_features = x
        fussion = []
        fuss_feature = []
        for num in range(self.i+1):
            feature_p = self.pool_layers[-(num+1)](enc_features[num])
            fussion.append(feature_p)
        concatenated_feature_map = torch.cat(fussion,dim=1)
        
            # fuss_feature.append(concatenated_feature_map)
        return concatenated_feature_map

    
class upsample(nn.Module):
    def __init__(
        self,
        i : int,
        ch : int,
        k: int,
    ) -> None:
        super().__init__()
        self.i = i
        self.up = nn.ModuleList()
        self.ch = ch
        spatial_dims = 3
        for num in range(3-i):
            in_channels = 2**(3-num)*ch
            out_channels = 2**k*ch
            s = in_channels//(ch*2**k)
            self.up.append(get_conv_layer(
                spatial_dims = 3,
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size= s,
                stride= s,
                conv_only=True,
                is_transposed=True,
            ))
        # print(self.up)
    def forward(self, x):
        enc_features = x
        fussion = []
        fuss_feature = []
        for num in range(3-self.i):
            feature_p = self.up[num](enc_features[-(num+1)])
            fussion.append(feature_p)
            # concatenated_feature_map = torch.cat(fussion,dim=1)
            # fuss_feature.append(concatenated_feature_map)
        concatenated_feature_map = torch.cat(fussion,dim=1)
        return concatenated_feature_map
    
# class skip_fussion(nn.Module):
#     def __init__(
#         self,
#         ch : int,
#     ) -> None:
#         super().__init__()
#         self.up=nn.ModuleList()
#         self.down=nn.ModuleList()
#         for num in range(3):
#             self.up.append(upsample(i=num,ch=ch,k=num))
#             self.down.append(Pool(i=num,ch=ch,k=num))    
#         self.cab1 = ConvAttn(ch=128, ch_out=32)
#         self.cab2 = ConvAttn(ch=224, ch_out=64)
#         self.cab3 = ConvAttn(ch=352, ch_out=128)

    
#     def _init_weights(self,m):
#         if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
#             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, (nn.InstanceNorm3d)):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
            
#     def forward(self, x):
#         cab = [self.cab1,self.cab2,self.cab3]
#         feature = x
#         skip_feature = []
#         for num in range(3):
#             up = self.up[num](x)
#             down = self.down[num](x)
#             skip = torch.cat([up,down], dim=1)
#             skip = cab[num](skip)
#             skip_feature.append(skip)
#         return skip_feature

class skip_fussion(nn.Module):
    def __init__(
        self,
        ch : int,
        i : int,
    ) -> None:
        super().__init__()
        self.i = i
#         self.up=nn.ModuleList()
#         self.down=nn.ModuleList()
        self.up=upsample(i=i,ch=ch,k=i)
        self.down=Pool(i=i,ch=ch,k=i)    
        self.cab1 = ConvAttn(ch=128, ch_out=32)
        self.cab2 = ConvAttn(ch=224, ch_out=64)
        self.cab3 = ConvAttn(ch=352, ch_out=128)

    
    def _init_weights(self,m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.InstanceNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        cab = [self.cab1,self.cab2,self.cab3]
        feature = x
        skip_feature = []
        up = self.up(x)
        down = self.down(x)
        skip = torch.cat([up,down], dim=1)
        skip = cab[self.i](skip)
        skip_feature.append(skip)
        return skip_feature[0]
        
class cat_out(nn.Module):
    def __init__(
        self,
        in_ch : int,
        in_ch2 : int,
        last_layer = False,
    ) -> None:
        super().__init__()
        if last_layer:
            self.up = get_conv_layer(3, in_ch2, in_ch, kernel_size=(2,4,4), stride=(2,4,4), conv_only=False, is_transposed=True)
        else:
            self.up = get_conv_layer(3, in_ch2, in_ch, kernel_size=(2,2,2), stride=(2,2,2), conv_only=False, is_transposed=True)
        
        self.norm = nn.InstanceNorm3d(in_ch)
        self.norm2 = nn.InstanceNorm3d(in_ch)
        # self.resconv = nn.Sequential(nn.Conv3d(in_ch,in_ch,kernel_size=3,stride=1,padding=1),
        #                              nn.InstanceNorm3d(in_ch),
        #                              nn.LeakyReLU(negative_slope=5e-2)
        #                           )
        
        self.resconv = get_conv_layer(3,in_ch + in_ch, in_ch, kernel_size=3, stride=1, conv_only=False)
        self.activ = nn.LeakyReLU(negative_slope=5e-2)
        
        if last_layer:
            self.upsample = nn.Upsample(scale_factor=(2,4,4),mode='trilinear',align_corners=True)
        else:
            self.upsample = nn.Upsample(scale_factor=(2,2,2),mode='trilinear',align_corners=True)
    def forward(self,x1, x2):
        x2_up = self.up(x2)
#         x = x1 + x2_up
        x = torch.cat([x1,x2_up],dim=1)
        output = self.resconv(x)
#         output = self.activ(output)

#         output = self.resconv(x)
        return output
        
    
        
    
if __name__=='__main__':
    enc4 = torch.randn((1,256,4,4,4))
    enc3 = torch.randn((1,128,8,8,8))
    enc2 = torch.randn((1,64,16,16,16))
    enc1 = torch.randn((1,32,32,32,32))
    model = skip_fussion(ch=32)
    x = [enc1,enc2,enc3,enc4]
    flop, para = profile(model, (x,))
    print(flop)
    print(para)
    print('flops: %.2f M, params: %.2f M' % (flop / 1000000.0, para / 1000000.0))
    s = skip_fussion(ch=32)
    skip = s(x)
    for i in range(3):
        print(skip[i].shape)
    