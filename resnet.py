import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.hub import load_state_dict_from_url
from typing import Type, Callable, Union, List, Optional, Any

# v1: Conv, BN, Relu, Relu after addition
# v2: full pre-activation: BN, Relu, Conv, Relu before addition

# a: Bottleneck places the stride at the first 1x1 convolution(self.conv1)
# b: Bottleneck places the stride for downsampling at 3x3 convolution(self.conv2)
# c: Based on b, c modifies the input stem
# d: Based on c, d modifies the downsampling block
# e: Based on d, e modifies the width of input stem
# s: Based on e, s uses the original downsampling block

__all__ = ['resnet18_v1', 'resnet34_v1', 'resnet50_v1', 'resnet101_v1', 'resnet152_v1',
          'resnet50_v1b', 'resnet101_v1b', 'resnet152_v1b',
          'resnet50_v1b_gn', 'resnet101_v1b_gn',
          'resnet50_v1c', 'resnet101_v1c', 'resnet152_v1c',
          'resnet50_v1d', 'resnet101_v1d', 'resnet152_v1d',
          'resnet50_v1e', 'resnet101_v1e', 'resnet152_v1e',
          'resnet50_v1s', 'resnet101_v1s', 'resnet152_v1s',
          'resnet18_v2', 'resnet34_v2', 'resnet50_v2', 'resnet101_v2', 'resnet152_v2'
          ]

model_urls = {
    'resnet18_v1': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet18_v1.pth',
    'resnet34_v1': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet34_v1.pth',
    'resnet50_v1': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet50_v1.pth',
    'resnet101_v1': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet101_v1.pth',
    'resnet152_v1': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet152_v1.pth',
    'resnet50_v1b': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet50_v1b.pth',
    'resnet101_v1b': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet101_v1b.pth',
    'resnet152_v1b': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet152_v1b.pth',
    'resnet50_v1b_gn': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet50_v1b_gn.pth',
    'resnet101_v1b_gn': '',
    'resnet50_v1c': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet50_v1c.pth',
    'resnet101_v1c': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet101_v1c.pth',
    'resnet152_v1c': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet152_v1c.pth',
    'resnet50_v1d': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet50_v1d.pth',
    'resnet101_v1d': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet101_v1d.pth',
    'resnet152_v1d': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet152_v1d.pth',
    'resnet50_v1e': '',
    'resnet101_v1e': '',
    'resnet152_v1e': '',
    'resnet50_v1s': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet50_v1s.pth',
    'resnet101_v1s': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet101_v1s.pth',
    'resnet152_v1s': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet152_v1s.pth',
    'resnet18_v2': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet18_v2.pth',
    'resnet34_v2': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet34_v2.pth',
    'resnet50_v2': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet50_v2.pth',
    'resnet101_v2': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet101_v2.pth',
    'resnet152_v2': 'https://github.com/liumoudong/ResNet-pytorch/releases/download/v1.0.0/resnet152_v2.pth',
}

resnet_layers = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
    'resnet152': [3, 8, 36, 3]
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class GroupNorm(nn.Module):
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(self, num_channels: int, num_groups: int = 32, eps: float = 1e-5, affine: bool = True) -> None:
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_channels))
            self.bias = Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.group_norm(
            input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)


class BasicBlockV1(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlockV1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class BasicBlockV2(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlockV2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + identity


class BottleneckV1(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BottleneckV1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, stride, bias=True)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, bias=True)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckV1b(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BottleneckV1b, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckV2(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BottleneckV2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(inplanes, width)
        self.bn2 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn3 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        return out + identity


class ResNetV1(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlockV1, BottleneckV1, BottleneckV1b]],
            layers: List[int],
            deep_stem=False,
            stem_width=32,
            avg_down=False,
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNetV1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.avg_down = avg_down
        if not deep_stem:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1,
                          bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, self.inplanes, kernel_size=3, stride=1, padding=1,
                          bias=False)
            )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, (BottleneckV1, BottleneckV1b)):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlockV1):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlockV1, BottleneckV1, BottleneckV1b]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride),
                    conv1x1(self.inplanes, planes * block.expansion, stride=1),
                    norm_layer(planes * block.expansion)
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion)
                )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ResNetV2(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlockV2, BottleneckV2]],
            layers: List[int],
            num_classes: int = 1000,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNetV2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.bn0 = norm_layer(3)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn_last = norm_layer(512 * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BasicBlockV2, BottleneckV2]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn_last(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet18_v1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BasicBlockV1, resnet_layers['resnet18'], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet18_v1'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet34_v1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BasicBlockV1, resnet_layers['resnet34'], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet34_v1'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50_v1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1, resnet_layers['resnet50'], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50_v1'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet101_v1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1, resnet_layers['resnet101'], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet101_v1'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet152_v1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1, resnet_layers['resnet152'], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet152_v1'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50_v1b(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1b, resnet_layers['resnet50'], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50_v1b'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50_v1b_gn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1b, resnet_layers['resnet50'], norm_layer=GroupNorm, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50_v1b_gn'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet101_v1b(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1b, resnet_layers['resnet101'], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet101_v1b'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet101_v1b_gn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1b, resnet_layers['resnet101'], norm_layer=GroupNorm, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['resnet101_v1b_gn'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet152_v1b(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1b, resnet_layers['resnet152'], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet152_v1b'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50_v1c(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1b, resnet_layers['resnet50'], deep_stem=True, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50_v1c'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet101_v1c(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1b, resnet_layers['resnet101'], deep_stem=True, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet101_v1c'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet152_v1c(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1b, resnet_layers['resnet152'], deep_stem=True, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet152_v1c'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50_v1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1b, resnet_layers['resnet50'], deep_stem=True, avg_down=True, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50_v1d'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet101_v1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1b, resnet_layers['resnet101'], deep_stem=True, avg_down=True, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet101_v1d'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet152_v1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1b, resnet_layers['resnet152'], deep_stem=True, avg_down=True, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet152_v1d'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50_v1e(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1b, resnet_layers['resnet50'], deep_stem=True, avg_down=True, stem_width=64, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['resnet50_v1e'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet101_v1e(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1b, resnet_layers['resnet101'], deep_stem=True, avg_down=True, stem_width=64, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['resnet101_v1e'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet152_v1e(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1b, resnet_layers['resnet152'], deep_stem=True, avg_down=True, stem_width=64, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['resnet152_v1e'],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet50_v1s(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1b, resnet_layers['resnet50'], deep_stem=True, stem_width=64, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50_v1s'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet101_v1s(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1b, resnet_layers['resnet101'], deep_stem=True, stem_width=64, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet101_v1s'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet152_v1s(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV1:
    model = ResNetV1(BottleneckV1b, resnet_layers['resnet152'], deep_stem=True, stem_width=64, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet152_v1s'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV2:
    model = ResNetV2(BasicBlockV2, resnet_layers['resnet18'], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet18_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet34_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV2:
    model = ResNetV2(BasicBlockV2, resnet_layers['resnet34'], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet34_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV2:
    model = ResNetV2(BottleneckV2, resnet_layers['resnet50'], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet101_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV2:
    model = ResNetV2(BottleneckV2, resnet_layers['resnet101'], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet101_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet152_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetV2:
    model = ResNetV2(BottleneckV2, resnet_layers['resnet152'], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet152_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
