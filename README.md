# ResNet-pytorch
ResNet, ResNet variants and pretrained weight files

**V1:** Conv, BN, Relu; Relu after addition <br />

**V2: **full pre-activation: BN, Relu, Conv; Relu before addition <br />



**orginal:** Bottleneck places the stride at the first 1x1 convolution(self.conv1) <br />
**b:** Bottleneck places the stride for downsampling at 3x3 convolution(self.conv2)<br />
**c: **Based on b, c modifies the input stem<br />
**d:** Based on c, d modifies the downsampling block<br />
**e:** Based on d, e modifies the width of input stem<br />
**s:** Based on e, s uses the original downsampling block<br />
