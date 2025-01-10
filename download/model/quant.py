from torch import Tensor
from mobilenet import *
from torch.nn.quantized import FloatFunctional
from torch.quantization import QuantStub, DeQuantStub
  
class QuantizableConvBN(ConvBN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(QuantizableConvBN, self).__init__(in_channels, out_channels, kernel_size, stride)
        self.float_functional = FloatFunctional()

    def fuse_model(self):
        fuse_modules(self.block, ['0', '1', '2'], inplace=True)


class QuantizableUniversalInvertedBottleneck(UniversalInvertedBottleneck):
    def __init__(self, *args, **kwargs):
        super(QuantizableUniversalInvertedBottleneck, self).__init__(*args, **kwargs)
        self.float_functional = FloatFunctional()

    def fuse_model(self):
        # Fuse start_dw_conv and start_dw_norm if start_dw_conv exists
        if self.start_dw_kernel_size:
            fuse_modules(self, ['start_dw_conv', 'start_dw_norm'], inplace=True)
        # Fuse expand_conv, expand_norm, and expand_act
        fuse_modules(self, ['expand_conv', 'expand_norm', 'expand_act'], inplace=True)
        # Fuse middle_dw_conv, middle_dw_norm, and middle_dw_act if middle_dw_conv exists
        if self.middle_dw_kernel_size:
            fuse_modules(self, ['middle_dw_conv', 'middle_dw_norm', 'middle_dw_act'], inplace=True)
        # Fuse proj_conv and proj_norm
        fuse_modules(self, ['proj_conv', 'proj_norm'], inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x

        if self.start_dw_kernel_size:
            x = self.start_dw_conv(x)
            x = self.start_dw_norm(x)

        x = self.expand_conv(x)
        x = self.expand_norm(x)
        x = self.expand_act(x)

        if self.middle_dw_kernel_size:
            x = self.middle_dw_conv(x)
            x = self.middle_dw_norm(x)
            x = self.middle_dw_act(x)

        x = self.proj_conv(x)
        x = self.proj_norm(x)

        if self.use_layer_scale:
            x = self.gamma * x

        if self.identity:
            return self.float_functional.add(x, shortcut)
        else:
            return x


class QuantizableMobileNetV4(nn.Module):
    DEFAULT_BLOCK_SPECS = [
        ('conv_bn', 3, 2, 32),
        ('conv_bn', 3, 2, 128),
        ('conv_bn', 1, 1, 48),
        # 3rd stage
        ('uib', 3, 5, 2, 80, 4.0),
        ('uib', 3, 3, 1, 80, 2.0),
        # 4th stage
        ('uib', 3, 5, 2, 160, 6.0),
        ('uib', 3, 3, 1, 160, 4.0),
        ('uib', 3, 3, 1, 160, 4.0),
        ('uib', 3, 5, 1, 160, 4.0),
        ('uib', 3, 3, 1, 160, 4.0),
        ('uib', 3, 0, 1, 160, 4.0),
        ('uib', 0, 0, 1, 160, 2.0),
        ('uib', 3, 0, 1, 160, 4.0),
        # 5th stage
        ('uib', 5, 5, 2, 256, 6.0),
        ('uib', 5, 5, 1, 256, 4.0),
        ('uib', 3, 5, 1, 256, 4.0),
        ('uib', 3, 5, 1, 256, 4.0),
        ('uib', 0, 0, 1, 256, 4.0),
        ('uib', 3, 0, 1, 256, 4.0),
        ('uib', 3, 5, 1, 256, 2.0),
        ('uib', 5, 5, 1, 256, 4.0),
        ('uib', 0, 0, 1, 256, 4.0),
        ('uib', 0, 0, 1, 256, 4.0),
        ('uib', 5, 0, 1, 256, 2.0),
        # FC layers
        ('conv_bn', 1, 1, 960),
    ]
    def __init__(self, block_specs=None, num_classes=1000):
        super(QuantizableMobileNetV4, self).__init__()

        if block_specs is None:
            block_specs = self.DEFAULT_BLOCK_SPECS

        c = 3
        layers = []
        for block_type, *block_cfg in block_specs:
            if block_type == 'conv_bn':
                block = QuantizableConvBN  
                k, s, f = block_cfg
                layers.append(block(c, f, k, s))
            elif block_type == 'uib':
                block = QuantizableUniversalInvertedBottleneck
                start_k, middle_k, s, f, e = block_cfg
                layers.append(block(c, f, e, start_k, middle_k, s))
            else:
                raise NotImplementedError
            c = f
        self.features = nn.Sequential(*layers)
        # Building last layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        hidden_channels = 1280
        self.conv = QuantizableConvBN(c, hidden_channels, 1) 
        self.classifier = nn.Linear(hidden_channels, num_classes)

        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        # Fuse all QuantizableConvBN and QuantizableUniversalInvertedBottleneck layers
        for module in self.features:
            if isinstance(module, QuantizableConvBN) or isinstance(module, QuantizableUniversalInvertedBottleneck):
                module.fuse_model()
        
        # Fuse the final ConvBN layer
        self.conv.fuse_model()