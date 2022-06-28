import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deepLabV3Plus.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.deepLabV3Plus.aspp import build_aspp
from models.deepLabV3Plus.decoder import build_decoder
from models.deepLabV3Plus.backbone import build_backbone
from config import LANDSAT8_META


class RGBSegNet(nn.Module):

    def __init__(self):
        super(RGBSegNet, self).__init__()
        self.deepLabV3 = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)

    def forward(self, x):
        # TODO: remove this
        r, g, b = LANDSAT8_META['rgb_bands']
        r, g, b = LANDSAT8_META['selected_bands'].index(r), LANDSAT8_META['selected_bands'].index(g), \
                  LANDSAT8_META['selected_bands'].index(b)
        x = torch.stack((x[:, r, :, :], x[:, g, :, :], x[:, b, :, :]), dim=1)

        out = self.deepLabV3(x)
        return out


class DeepLabV3PlusRGB(nn.Module):

    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLabV3PlusRGB, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        self.input_depth = len(LANDSAT8_META['selected_bands'])  # number of input channels
        self.backbone = build_backbone(backbone, output_stride, BatchNorm, pretrained=False,
                                       input_depth=self.input_depth)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input_):
        # r, g, b = LANDSAT8_META['rgb_bands']
        # r, g, b = LANDSAT8_META['selected_bands'].index(r), LANDSAT8_META['selected_bands'].index(g), \
        #           LANDSAT8_META['selected_bands'].index(b)
        # rgb_input = torch.stack((input_[:, r, :, :], input_[:, g, :, :], input_[:, b, :, :]), dim=1)

        x, low_level_feat = self.backbone(input_)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input_.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p