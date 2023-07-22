import torch
import torch.nn as nn

'''
each line means...
     # Tuple: Conv layer (kernel_size, output_channel, stride, padding)
     # "M": Maxpool layer
     # List: Conv layer repeats [(Conv_layer), (Conv_layer), ..., 반복횟수 ] 즉, 반복되는 블록 묶은것
'''
architecture_config = [
     (7, 64, 2, 3),
     "M",
     (3, 192, 1, 1),
     "M",
     
     (1, 128, 1, 0),
     (3, 256, 1, 1),
     (1, 256, 1, 0),
     (3, 512, 1, 1),
     "M",
     
     [(1, 256, 1, 0), (3, 512, 1, 1), 4],
     (1, 512, 1, 0),
     (3, 1024, 1, 1),
     "M",
     
     [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
     (3, 1024, 1, 1),
     (3, 1024, 2, 1),
     
     (3, 1024, 1, 1),
     (3, 1024, 1, 1),
]


class CNNBlock(nn.Module): # 여러 번 사용할 예정
     
     def __init__(self, in_channels, out_channels, **kwargs): # keyword argument 사용할 에정
          super(CNNBlock, self).__init__()
          
          # bias=False 이유: batch norm 사용할 것이기 때문 (참고: 논문 게재 당시에는 batch norm이 도입되지 않음)
          self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
          self.batchnorm = nn.BatchNorm2d(out_channels)
          self.leakyrelu = nn.LeakyReLU(0.1) # 0.1: 학습률, 기울기 의미
     
     def forward(self, x):
          return self.leakyrelu(self.batchnorm(self.conv(x)))
     
class YOLOv1(nn.Module):
     def __init__(self, in_channels=3, **kwargs):
          super(YOLOv1, self).__init__()
          
          self.architecture = architecture_config
          self.in_channels = in_channels
          self.darknet = self._create_conv_layers(self.architecture)
          self.fcs = self._create_fcs(**kwargs)
          
     def forward(self, x):
          x = self.darknet(x)
          return self.fcs(torch.flatten(x, start_dim=1))
     
     # 여기서 darknet 만들거임
     def _create_conv_layers(self, architecture):
          layers = []     # layer list 받을 예정
          in_channels = self.in_channels
          
          # 여기서 레이어 하나씩 만들어줌
          for x in architecture:
               if type(x) == tuple:
                    layers += [
                         CNNBlock(
                              in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3]),
                         ]
                    in_channels = x[1]
                    
               elif type(x) == str:
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
               elif type(x) == list:
                    
                    # 리스트 예시: [(1, 512, 1, 0), (3, 1024, 1, 1), 2]
                    
                    # 강연자 방법
                    conv1 = x[0]
                    conv2 = x[1]
                    num_repeats = x[2]
                    for _ in range(num_repeats):
                         layers += [CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]), ]
                         layers += [CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]), ]
                    
                    in_channels = conv2[1]
                         
                    # 내 방법
                    # num_repeats = x[-1]
                    # for _ in range(num_repeats):
                    #      for sub_x in x[:-1]:
                    #           layers += [ CNNBlock(in_channels, sub_x[1], kernel_size=sub_x[0], stride=sub_x[2], padding=sub_x[3]), ]
                    #           in_channels = sub_x[0]
                    
          return nn.Sequential(*layers) # list 형태로 반환 (이를 unpack 상태로 반환 -> 추후 할당 받는 곳에서 nn.Sequential 형태로 변환함
          
     def _create_fcs(self, split_size, num_boxes, num_classes):
          S, B, C = split_size, num_boxes, num_classes
          return nn.Sequential(
               nn.Flatten(),
               nn.Linear(1024 * S * S, 496),      # 논문에서는 out_features=4096
               nn.Dropout(0.0),
               nn.LeakyReLU(0.1),
               nn.Linear(496, S * S * (C + B * 5)),      # (S, S, 30)
          )
          
          
def test(S=7, B=2, C=20):
     model = YOLOv1(split_size=S, num_boxes=B, num_classes=C)
     x = torch.randn(size=(2, 3, 448, 448))
     print(model(x).shape)
     
test()