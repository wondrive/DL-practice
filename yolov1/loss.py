import torch
import torch.nn as nn
from utils import intersection_over_union    # 추후 생성할 모듈

class YoloLoss(nn.Module):
     def __init__(self, S=7, B=2, C=20):
          super(YoloLoss, self).__init__()
          self.mse = nn.MSELoss(reduction="sum")  # 각 항의 덧셈 기준으로 손실 최소화reduction 할거임 (평균.. 등이 아니라)
          self.S = S
          self.B = B
          self.C = C
          self.lambda_noobj = 0.5
          self.lambda_coord = 5
          
     def forward(self, predictions, target):
          predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)   # 최종 예측 tensor 형태
          
          iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])   # 첫 번째 bbox의 IOU 면적
          iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
          ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)        # [b1, b2] 합침
          ious_maxes, best_box = torch.max(ious, dim=0) # IOU 더 큰 box의 [IOU, index] 반환
          exists_box = target[..., 20].unsqueeze(3)    # identity obj_i, 즉 해당 셀의 conf_score
          
          # =================== #
          # For BOX COORDINATES #
          # =================== #
          # 셀에 객체 있다면, 두 개 박스 중 best_box만 살리는 코드 (b1 or b2)
          box_predictions = exists_box * (
               best_box * predictions[..., 26:30]
               + (1 - best_box) * predictions[..., 21:25]
          )
          box_targets = exists_box * target[..., 21:25]
          
          # x, y 좌표
          box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt( # 기울기 부호가 정확하도록 해주기 위해서 torch.sign(부호) 곱해주기 (Q. 질문: 근데 왜 절대값으로 해줬다가 다시 부호를 붙여줬을까??)
               torch.abs(box_predictions[..., 2:4] + 1e-6)  # 1e-6는 논문에는 없지만 값이 0일때 기울기 발산하므로 더해줌.. 근데 잘 모르겠음
          )
          
          box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
          
          # Input shape: (N, S, S, 4) -> (N*S*S, 4)
          box_loss = self.mse(
               torch.flatten(box_predictions, end_dim=-2),  # 뒤에서 두번째 차원 까지만 평평하게
               torch.flatten(box_targets, end_dim=-2)
          )
          
          # =================== #
          #   For OBJECT LOSS   #
          # =================== #
          pred_box = (
               best_box * predictions[..., 25:26]
               + (1 - best_box) * predictions[..., 20:21]
          )
          
          # Input shape: (N*S*S)
          object_loss = self.mse(
               torch.flatten(exists_box * pred_box),
               torch.flatten(exists_box * target[..., 20:21])
          )
          
          # =================== #
          #  For NO OBJECT LOSS #
          # =================== #
          # (N, S, S, 1) -> (N, S*S)
          no_object_loss = self.mse(
               torch.flatten((1-exists_box) * predictions[..., 20:21], start_dim=1),
               torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1),
          )
          
          no_object_loss += self.mse(
               torch.flatten((1-exists_box) * predictions[..., 25:56], start_dim=1),
               torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1),
          )
          
          
          # =================== #
          #    For CLASS lOSS   #
          # =================== #
          # (N, S, S, 20) -> (N*S*S, 20)
          class_loss = self.mse(
               torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
               torch.flatten(exists_box * target[..., :20], end_dim=-2),
          )
          
          
          # 최종 loss들 합
          loss = (
               self.lambda_coord * box_loss
               + object_loss
               + self.lambda_noobj * no_object_loss
               + class_loss
          )
          
          return loss