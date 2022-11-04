# ResNet-pytorch
ResNet, ResNet variants and pretrained weight files

v1: Conv, BN, Relu, Relu after addition
v2: full pre-activation: BN, Relu, Conv, Relu before addition

orginal: Bottleneck places the stride at the first 1x1 convolution(self.conv1)
b: Bottleneck places the stride for downsampling at 3x3 convolution(self.conv2)
c: Based on b, c modifies the input stem
d: Based on c, d modifies the downsampling block
e: Based on d, e modifies the width of input stem
s: Based on e, s uses the original downsampling block
