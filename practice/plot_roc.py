import torch
import matplotlib.pyplot as plt
from collections import Counter


def plot_roc(pred_boxes, true_boxes, num_classes, iou_threshold=0.5, box_format="corners"):
     
     # 본인 클래스로 수정
     classes = {0:'사과', 1:'포도', 2:'배'}
     
     AUCs = []
     TPRs = []
     FPRs = []

     epsilon = 1e-6      # 추후 수지적 안정을 위해 사용

     for c in range(num_classes):
          detections = []
          ground_truths = []
          
          for detection in pred_boxes:
               if detection[1] == c:       # class == c
                    detections.append(detection)

          for true_box in true_boxes:
               if true_box[1] == c:
                    ground_truths.append(true_box)
          
          
          # gt[0] : 각 이미지별 gt 갯수 세어서 dict 형태로 저장 (class==c)
          # 예) amount_bboxes = {클래스0:3개, 1:5. 2:7}
          amount_bboxes = Counter([gt[0] for gt in ground_truths])
          
          # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
          for key, val in amount_bboxes.items():
               amount_bboxes[key] = torch.zeros(val)

          # confidence score 기준 내림차순 정렬
          detections.sort(key=lambda x: x[2], reverse=True)
          TP = torch.zeros((len(detections)))
          FP = torch.zeros((len(detections)))
          total_true_bboxes = len(ground_truths)
          
          # continue 이유: TP=0이라 recall은 분모가 0이 되어서 계산불가
          if total_true_bboxes == 0:
               continue
          
          for detection_idx, detection in enumerate(detections):
               
               # 해당 이미지의 gt 불러옴
               ground_truth_img = [
                    bbox for bbox in ground_truths if bbox[0] == detection[0]
               ]

               num_gts = len(ground_truth_img)
               best_iou = 0

               for idx, gt in enumerate(ground_truth_img):
                    '''
                    intersection_over_union() 함수는 다음 github 오픈소스 참고
                    github @aladdinpersson : https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/metrics/iou.py
                    '''
                    iou = intersection_over_union(
                         torch.tensor(detection[3:]),
                         torch.tensor(gt[3:]),
                         box_format=box_format,
                    )
                    
                    if iou > best_iou:
                         best_iou = iou
                         best_gt_idx = idx

               if best_iou > iou_threshold:
                    # (참고) ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
                    # amount_bboxes[ train_idx ][ 객체 idx ]
                    # 해당 이미지의, 해당 객체 발견하면 1
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                         TP[detection_idx] = 1
                         amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                         FP[detection_idx] = 1
               else:
                    FP[detection_idx] = 1

          TP_cumsum = torch.cumsum(TP, dim=0)     # 누적합
          FP_cumsum = torch.cumsum(FP, dim=0)
          TN_cumsum = FP_cumsum[-1] - FP_cumsum
          
          TPR = TP_cumsum / (total_true_bboxes + epsilon) # sensitivity
          FPR = torch.div(FP_cumsum, (TN_cumsum + FP_cumsum + epsilon)) # 1 - specificity
          TPRs.append(TPR)
          FPRs.append(FPR)
          AUCs.append(torch.trapz(TPR, FPR))  # trapz: 적분 (AUC구해줌)
          
     # plot ROC curves
     plt.figure(figsize=(15, 5))
     for i in range(num_classes):
          plt.subplot(131+i)
          plt.plot(FPRs[i], TPRs[i], color='red', label=classes[i])
          plt.plot([0, 1], [0, 1], color='black', linestyle='--')
          plt.xlim([0.0, 1.0])
          plt.ylim([0.0, 1.05])
          plt.xlabel('FPR')
          plt.ylabel('TPR')
          plt.title(f'ROC curve ({classes[i]})\nAUC: {AUCs[i]:.2f}')
          plt.legend(loc='lower right')
     plt.show()
