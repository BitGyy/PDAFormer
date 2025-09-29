import torch
from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from unetr_pp.network_architecture.unetr_pp.layers import LayerNorm
from unetr_pp.network_architecture.unetr_pp.transformerblock import TransformerBlock,NextBlock
# from unetr_pp.network_architecture.unetr_pp.transformerblock import TransformerBlock,NextBlock, NextBlock_ca
from unetr_pp.network_architecture.unetr_pp.dynunet_block import get_conv_layer, UnetResBlock,MHCResBlock


einops, _ = optional_import("einops")

class UnetrPPEncoder(nn.Module):
    def __init__(self, input_size=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4],dims=[32, 64, 128, 256],
                 proj_size =[256,128,64,32], depths=[3, 3, 3, 3],  num_heads=4, spatial_dims=3, in_channels=1, dropout=0.0, transformer_dropout_rate=0.1 ,**kwargs):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(2, 4, 4), stride=(2, 4, 4),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(TransformerBlock(input_size=input_size[i], hidden_size=dims[i],  proj_size=proj_size[i], num_heads=num_heads,
                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states
    
class NextEncoder(nn.Module):
    # def __init__(self, input_size=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4],insize=[32,16,8,4],dims=[32, 64, 128, 256],
    #              proj_size =[256,128,64,32], depths=[3, 3, 3, 3],  num_heads=[4, 4, 4, 4], spatial_dims=3, in_channels=1, dropout=0.0, transformer_dropout_rate=0.1 ,**kwargs):
    #     super().__init__()
    def __init__(self, input_size=[16 * 40 * 40, 8 * 20 * 20, 4 * 10 * 10, 2 * 5 * 5],insize=[32,16,8,4],dims=[32, 64, 128, 256],
                 proj_size =[256,128,64,32], depths=[3, 3, 3, 3],  num_heads=[4, 4, 4, 4], spatial_dims=3, in_channels=1, dropout=0.0, transformer_dropout_rate=0.1 ,**kwargs):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(1, 4, 4), stride=(1, 4, 4),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            # if i==0:
            #     for j in range(depths[0]):
            #         stage_blocks.append(MHCResBlock(spatial_dims, dims[i], dims[i], kernel_size=3, stride=1,
            #                  norm_name="instance",groups=dims[i]//32))
            #         stage_blocks.append(UnetResBlock(spatial_dims, dims[i], dims[i], kernel_size=3, stride=1,norm_name="instance",groups=dims[i]))
            #     # self.stages.append(nn.Sequential(*stage_blocks))
            # else:
            #     for j in range(depths[i]-1):
            #         stage_blocks.append(MHCResBlock(spatial_dims, dims[i], dims[i], kernel_size=3, stride=1, norm_name="instance",groups=dims[i]//32))
            #         stage_blocks.append(UnetResBlock(spatial_dims, dims[i], dims[i], kernel_size=3, stride=1,norm_name="instance"))
            #         stage_blocks.append(nn.Dropout3d(0.1, False))
            #     stage_blocks.append(NextBlock(input_size=input_size[i], hidden_size=dims[i],  proj_size=proj_size[i], num_heads=num_heads,
            #                              dropout_rate=transformer_dropout_rate, pos_embed=True))

            for j in range(depths[i]):
                stage_blocks.append(NextBlock(input_size=input_size[i] ,hidden_size=dims[i],  proj_size=proj_size[i], num_heads=num_heads[i],
                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))

        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            

    def forward_features(self, x):
        hidden_states = []
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        # print(x.shape)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states
    
class NextEncoder_ca(nn.Module):
    def __init__(self, input_size=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4],insize=[32,16,8,4],dims=[32, 64, 128, 256],
                 proj_size =[256,128,64,32], depths=[3, 3, 3, 3],  num_heads=4, spatial_dims=3, in_channels=1, dropout=0.0, transformer_dropout_rate=0.1 ,**kwargs):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(2, 4, 4), stride=(2, 4, 4),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            # stage_blocks.append()
            # if i==0:
            #     for j in range(depths[0]):
            #         stage_blocks.append(MHCResBlock(spatial_dims, dims[i], dims[i], kernel_size=3, stride=1,
            #                  norm_name="instance",groups=dims[i]//32))
            #         stage_blocks.append(UnetResBlock(spatial_dims, dims[i], dims[i], kernel_size=3, stride=1,norm_name="instance",groups=dims[i]))
            #     # self.stages.append(nn.Sequential(*stage_blocks))
            # else:
            #     for j in range(depths[i]-1):
            #         stage_blocks.append(MHCResBlock(spatial_dims, dims[i], dims[i], kernel_size=3, stride=1, norm_name="instance",groups=dims[i]//32))
            #         stage_blocks.append(UnetResBlock(spatial_dims, dims[i], dims[i], kernel_size=3, stride=1,norm_name="instance"))
            #         stage_blocks.append(nn.Dropout3d(0.1, False))
            #     stage_blocks.append(NextBlock(input_size=input_size[i], hidden_size=dims[i],  proj_size=proj_size[i], num_heads=num_heads,
            #                              dropout_rate=transformer_dropout_rate, pos_embed=True))
            for j in range(depths[i]):
                stage_blocks.append(NextBlock(input_size=input_size[i] ,hidden_size=dims[i],  proj_size=proj_size[i], num_heads=num_heads,
                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            

    def forward_features(self, x):
        hidden_states = []
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        # print(x.shape)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states


class UnetrUpBlock(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size= out_channels, proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):

        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)

        return out
                                                     
class MHCUpBlock(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(MHCResBlock(spatial_dims, out_channels, out_channels, kernel_size=3, stride=1,
                             norm_name=norm_name,groups=out_channels //32))
                stage_blocks.append(UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=3, stride=1,
                             norm_name=norm_name))
            # stage_blocks.append(NextBlock(input_size=out_size, hidden_size=out_channels,  proj_size=proj_size, num_heads=num_heads,
            #                          dropout_rate=0.1, pos_embed=True))
            # for j in range(depth):
            #     stage_blocks.append(NextBlock(input_size=out_size, hidden_size= out_channels, proj_size=proj_size, num_heads=num_heads,
            #                                          dropout_rate=0.1, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):

        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)

        return out   

class NextUpBlock(nn.Module):
    def     __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            # for j in range(depth-1):
            #     stage_blocks.append(MHCResBlock(spatial_dims, out_channels, out_channels, kernel_size=3, stride=1,
            #                  norm_name=norm_name,groups=out_channels //32))
            #     stage_blocks.append(UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=3, stride=1,
            #                  norm_name=norm_name))
            # stage_blocks.append(NextBlock(input_size=out_size, hidden_size=out_channels,  proj_size=proj_size, num_heads=num_heads,
            #                          dropout_rate=0.1, pos_embed=True))
            for j in range(depth),:
                stage_blocks.append(NextBlock(input_size=out_size, hidden_size= out_channels, proj_size=proj_size, num_heads=num_heads,
                                                     dropout_rate=0.1, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))
        
        
            
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)): 
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, inp, skip):
        
        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)

        return out  

    
class mp_ap(nn.Module):
    def __init__(
        self,
        i : int,
    ) -> None:
        super().__init__()
        self.maxpool = nn.MaxPool3d(kernel_size = 2**(i+1), stride = 2**(i+1))
        self.avgpool = nn.AvgPool3d(kernel_size = 2**(i+1), stride = 2**(i+1))
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)): 
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        output = self.maxpool(x) + self.avgpool(x)
        return output

    
class upsample(nn.Module):
    def __init__(
        self,
        i : int,
        ch : int,
    ) -> None:
        super().__init__()
        self.up = nn.ModuleList()
        self.ch = ch
        spatial_dims = 3
        for num in range(4-i):  
            self.up.append(get_conv_layer(
                spatial_dims = 3,
                in_channels = 2**num*ch,
                out_channels = 2**num*ch,
                kernel_size= 2**num,
                stride= 2**num,
                conv_only=True,
                is_transposed=True,
            ))
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)): 
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        output = self.up(x)
        return output
    
    
class Pool(nn.Module):
    def __init__(
        self,
        i: int,
        ch: int,
    ) -> None:
        super().__init__()
        self.i = i
        self.ch = ch
        self.up_sample = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        for num in range(i-1):
            self.pool_layers.append(mp_ap(num))
            self.conv_blocks.append()
        
    def forward(self, enc1, enc2, enc3, enc4):
        enc_features = [enc1, enc2, enc3, enc4]
        fussion = []
        for num in range(self.i-1):
            feature_p = self.pool_layers[-num](enc_features[num])
            fussion.append(feature_p)
        fussion.append(enc_features[self.i-1])
        concatenated_feature_map = torch.cat(fussion,dim=1)
        
        return concatenated_feature_map
    
class ConvAttn(nn.Module):
    def __init__(
        self,
        i,
        channel,
    ) -> None:
        super().__init__()
        self.i = i
        self.channel = channel
        in_ch = 0
        for num in range(i):
            in_ch = in_ch + channel*2**num
        out_ch = 2**i*channel
        self.c1 = nn.Conv3d(in_channels = in_ch, out_channels = out_ch, kernel_size = 3, stride = 1, padding = 1)
        self.c2 = nn.Conv3d(in_channels = out_ch, out_channels = out_ch, kernel_size = 3, stride = 1, padding = 1)
        self.norm = nn.BatchNorm3d(out_ch)
        self.activate = nn.LeakyReLU(negative_slope=5e-2)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)): 
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self,x):
        x_1 = self.activate(self.norm(self.c1(x)))
        x_2 = self.norm(self.c2(x_1))
        output = self.activate(x + x_2)
        return output
        
    
    
    
# class MultiScaleFeatureFussion(nn.Module):
#     def __init__(
#         self,
#         i:int,
#         img_size:int,
#         ch:int,
#     ) -> None:
#         super().__init__()
        
    
#original NetxUpBlock
# class NextUpBlock(nn.Module):
#     def     __init__(
#             self,
#             spatial_dims: int,
#             in_channels: int,
#             out_channels: int,
#             kernel_size: Union[Sequence[int], int],
#             upsample_kernel_size: Union[Sequence[int], int],
#             norm_name: Union[Tuple, str],
#             proj_size: int = 64,
#             num_heads: int = 4,
#             out_size: int = 0,
#             depth: int = 3,
#             conv_decoder: bool = False,
#     ) -> None:
#         """
#         Args:
#             spatial_dims: number of spatial dimensions.
#             in_channels: number of input channels.
#             out_channels: number of output channels.
#             kernel_size: convolution kernel size.
#             upsample_kernel_size: convolution kernel size for transposed convolution layers.
#             norm_name: feature normalization type and arguments.
#             proj_size: projection size for keys and values in the spatial attention module.
#             num_heads: number of heads inside each EPA module.
#             out_size: spatial size for each decoder.
#             depth: number of blocks for the current decoder stage.
#         """

#         super().__init__()
#         upsample_stride = upsample_kernel_size
#         self.transp_conv = get_conv_layer(
#             spatial_dims,
#             in_channels,
#             out_channels,
#             kernel_size=upsample_kernel_size,
#             stride=upsample_stride,
#             conv_only=True,
#             is_transposed=True,
#         )

#         # 4 feature resolution stages, each consisting of multiple residual blocks
#         self.decoder_block = nn.ModuleList()

#         # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block (see suppl. material in the paper)
#         if conv_decoder == True:
#             self.decoder_block.append(
#                 UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
#                              norm_name=norm_name, ))
#         else:
#             stage_blocks = []
#             # for j in range(depth-1):
#             #     stage_blocks.append(MHCResBlock(spatial_dims, out_channels, out_channels, kernel_size=3, stride=1,
#             #                  norm_name=norm_name,groups=out_channels //32))
#             #     stage_blocks.append(UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=3, stride=1,
#             #                  norm_name=norm_name))
#             # stage_blocks.append(NextBlock(input_size=out_size, hidden_size=out_channels,  proj_size=proj_size, num_heads=num_heads,
#             #                          dropout_rate=0.1, pos_embed=True))
#             for j in range(depth),:
#                 stage_blocks.append(NextBlock(input_size=out_size, hidden_size= out_channels, proj_size=proj_size, num_heads=num_heads,
#                                                      dropout_rate=0.1, pos_embed=True))
#             self.decoder_block.append(nn.Sequential(*stage_blocks))

#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, (nn.LayerNorm)):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self, inp, skip):

#         out = self.transp_conv(inp)
#         out = out + skip
#         out = self.decoder_block[0](out)

#         return out

if __name__=='__main__':
    j = upsample(i=2, ch=32)
    print(j)
    
    