from models.deepLabV3Plus.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm, pretrained, input_depth):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm, pretrained=pretrained, input_depth=input_depth)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm, pretrained=pretrained)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm, pretrained=pretrained)
    else:
        raise NotImplementedError
