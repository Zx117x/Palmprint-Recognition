import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo


__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):
# Fire类继承nn.Module类
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes# 输入通道数
        # squeeze layer，由二维1∗1卷积组成,激活函数为ReLU，inplace=True原地操作减少内存
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        # expandlayer中的1*1卷积
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)    
        self.expand1x1_activation = nn.ReLU(inplace=True)
        
        # expandlayer中的3*3卷积
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        #将输入x经过squeeze layer进行卷积操作
        x = self.squeeze_activation(self.squeeze(x))
        #分别送入expand1x1和expand3x3进行卷积
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)
       #两个维度相同的输出张量连接在一起。注意这里的dim=1，即按照列连接，最终得到若干行。


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()#调用基类初始化函数对基类进行初始化。
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential( #Sequential为序列容器,按照顺序将Modules添加到其中
                nn.Conv2d(3, 96, kernel_size=7, stride=2),#Conv1: 输入3通道，输出96通道
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),#ceil_mode向上取整
                Fire(96, 16, 64, 64),#Fire2: 输入96通道，输出两个64通道
                Fire(128, 16, 64, 64),#Fire3: 输入128通道，输出两个64通道
                Fire(128, 32, 128, 128),#Fire4: 输入128通道，输出两个128通道
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),#ceil_mode向上取整
                Fire(256, 32, 128, 128),#Fire5: 输入256通道，输出两个128通道
                Fire(256, 48, 192, 192),#Fire6: 输入256通道，输出两个192通道
                Fire(384, 48, 192, 192),#Fire7: 输入384通道，输出两个192通道
                Fire(384, 64, 256, 256),#Fire8: 输入384通道，输出两个256通道
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),#ceil_mode向上取整
                Fire(512, 64, 256, 256),#Fire9: 输入512通道，输出两个256通道
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:#最后一层使用了均值为0，方差为0.01的正太分布初始化方法
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:#其余层使用He Kaiming论文中的均匀分布初始化方法
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)#偏置初始化为0

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes) #输出拉伸为一维映射至num_classes个类别


def squeezenet1_0(pretrained=False, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.0, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_0']))
    return model


def squeezenet1_1(pretrained=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.1, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']))
    return model
