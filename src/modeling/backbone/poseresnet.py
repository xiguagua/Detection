from detectron2.modeling.backbone import Backbone

import torch.nn as nn

BN_MOMENTUM = 0.1

class PoseResNet(Backbone):
  def __init__(self, block, layers, heads, head_conv, **kwargs):
    self.inplanes = 64
    self.deconv_with_bias = False
    self.heads = heads

    super().__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
    self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    # used for deconv layers
    self.deconv_layers = self._make_deconv_layer(
        3,
        [256, 256, 256],
        [4, 4, 4],
    )
    # self.final_layer = []

    # for head in sorted(self.heads):
    #   num_output = self.heads[head]
    #   if head_conv > 0:
    #     fc = nn.Sequential(
    #         nn.Conv2d(256, head_conv,
    #           kernel_size=3, padding=1, bias=True),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(head_conv, num_output, 
    #           kernel_size=1, stride=1, padding=0))
    #   else:
    #     fc = nn.Conv2d(
    #       in_channels=256,
    #       out_channels=num_output,
    #       kernel_size=1,
    #       stride=1,
    #       padding=0
    #   )
    #   self.__setattr__(head, fc)

    # self.final_layer = nn.ModuleList(self.final_layer)

  def _make_layer(self, block, planes, blocks, stride=1):
      downsample = None
      if stride != 1 or self.inplanes != planes * block.expansion:
          downsample = nn.Sequential(
              nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
              nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
          )

      layers = []
      layers.append(block(self.inplanes, planes, stride, downsample))
      self.inplanes = planes * block.expansion
      for i in range(1, blocks):
          layers.append(block(self.inplanes, planes))

      return nn.Sequential(*layers)

  def _get_deconv_cfg(self, deconv_kernel, index):
    padding, output_padding = {
        4: (1, 0),
        3: (1, 1),
        2: (0, 0),
    }[deconv_kernel]
    
    return deconv_kernel, padding, output_padding

  def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
      assert num_layers == len(num_filters), \
          'ERROR: num_deconv_layers is different len(num_deconv_filters)'
      assert num_layers == len(num_kernels), \
          'ERROR: num_deconv_layers is different len(num_deconv_filters)'

      layers = []
      for i in range(num_layers):
          kernel, padding, output_padding = \
              self._get_deconv_cfg(num_kernels[i], i)

          planes = num_filters[i]
          layers.append(
              nn.ConvTranspose2d(
                  in_channels=self.inplanes,
                  out_channels=planes,
                  kernel_size=kernel,
                  stride=2,
                  padding=padding,
                  output_padding=output_padding,
                  bias=self.deconv_with_bias))
          layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
          layers.append(nn.ReLU(inplace=True))
          self.inplanes = planes

      return nn.Sequential(*layers)

  def forward(self, x):
    """
    Returns:
      dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
    """
    assert x.dim() == 4, f"PoseResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
    outputs = {}
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.deconv_layers(x)
    outputs['deconv'] = x    
    # for head in self.heads:
    #     ret[head] = self.__getattr__(head)(x)
    return outputs

  def init_weights(self, num_layers, pretrained=True):
      if pretrained:
          # print('=> init resnet deconv weights from normal distribution')
          for _, m in self.deconv_layers.named_modules():
              if isinstance(m, nn.ConvTranspose2d):
                  # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                  # print('=> init {}.bias as 0'.format(name))
                  nn.init.normal_(m.weight, std=0.001)
                  if self.deconv_with_bias:
                      nn.init.constant_(m.bias, 0)
              elif isinstance(m, nn.BatchNorm2d):
                  # print('=> init {}.weight as 1'.format(name))
                  # print('=> init {}.bias as 0'.format(name))
                  nn.init.constant_(m.weight, 1)
                  nn.init.constant_(m.bias, 0)
          # print('=> init final conv weights from normal distribution')
          for head in self.heads:
            final_layer = self.__getattr__(head)
            for i, m in enumerate(final_layer.modules()):
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    if m.weight.shape[0] == self.heads[head]:
                        if 'hm' in head:
                            nn.init.constant_(m.bias, -2.19)
                        else:
                            nn.init.normal_(m.weight, std=0.001)
                            nn.init.constant_(m.bias, 0)
          #pretrained_state_dict = torch.load(pretrained)
          url = model_urls['resnet{}'.format(num_layers)]
          pretrained_state_dict = model_zoo.load_url(url)
          print('=> loading pretrained model {}'.format(url))
          self.load_state_dict(pretrained_state_dict, strict=False)
      else:
          print('=> imagenet pretrained model dose not exist')
          print('=> please download it first')
          raise ValueError('imagenet pretrained model does not exist')
