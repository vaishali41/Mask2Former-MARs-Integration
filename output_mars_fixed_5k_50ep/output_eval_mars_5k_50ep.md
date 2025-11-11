index created!
[10/22 15:23:18 d2.evaluation.fast_eval_api]: Evaluate annotation type *segm*
[10/22 15:23:19 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 1.71 seconds.
[10/22 15:23:20 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[10/22 15:23:20 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 0.16 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
[10/22 15:23:20 d2.evaluation.coco_evaluation]: Evaluation results for segm: 
|  AP   |  AP50  |  AP75  |  APs  |  APm  |  APl  |
|:-----:|:------:|:------:|:-----:|:-----:|:-----:|
| 0.000 | 0.000  | 0.000  | 0.000 | 0.000 | 0.000 |
[10/22 15:23:20 d2.evaluation.coco_evaluation]: Per-category segm AP: 
| category      | AP    | category     | AP    | category       | AP    |
|:--------------|:------|:-------------|:------|:---------------|:------|
| person        | 0.000 | bicycle      | 0.000 | car            | 0.000 |
| motorcycle    | 0.000 | airplane     | 0.000 | bus            | 0.000 |
| train         | 0.000 | truck        | 0.000 | boat           | 0.000 |
| traffic light | 0.000 | fire hydrant | 0.000 | stop sign      | 0.000 |
| parking meter | 0.000 | bench        | 0.000 | bird           | 0.000 |
| cat           | 0.000 | dog          | 0.000 | horse          | 0.000 |
| sheep         | 0.000 | cow          | 0.000 | elephant       | 0.000 |
| bear          | 0.000 | zebra        | 0.000 | giraffe        | 0.000 |
| backpack      | 0.000 | umbrella     | 0.000 | handbag        | 0.000 |
| tie           | 0.000 | suitcase     | 0.000 | frisbee        | 0.000 |
| skis          | 0.000 | snowboard    | 0.000 | sports ball    | 0.000 |
| kite          | 0.000 | baseball bat | 0.000 | baseball glove | 0.000 |
| skateboard    | 0.000 | surfboard    | 0.000 | tennis racket  | 0.000 |
| bottle        | 0.000 | wine glass   | 0.000 | cup            | 0.000 |
| fork          | 0.000 | knife        | 0.000 | spoon          | 0.000 |
| bowl          | 0.000 | banana       | 0.000 | apple          | 0.000 |
| sandwich      | 0.000 | orange       | 0.000 | broccoli       | 0.000 |
| carrot        | 0.000 | hot dog      | 0.000 | pizza          | 0.000 |
| donut         | 0.000 | cake         | 0.000 | chair          | 0.001 |
| couch         | 0.000 | potted plant | 0.000 | bed            | 0.000 |
| dining table  | 0.000 | toilet       | 0.000 | tv             | 0.000 |
| laptop        | 0.000 | mouse        | 0.000 | remote         | 0.000 |
| keyboard      | 0.000 | cell phone   | 0.000 | microwave      | 0.000 |
| oven          | 0.000 | toaster      | 0.000 | sink           | 0.000 |
| refrigerator  | 0.000 | book         | 0.000 | clock          | 0.000 |
| vase          | 0.000 | scissors     | 0.000 | teddy bear     | 0.000 |
| hair drier    | nan   | toothbrush   | 0.000 |                |       |

--- Image 0 ---
Total predictions: 100
Score stats:
  Min: 0.0000
  Max: 0.1720
  Mean: 0.0153
  Above 0.5: 0
  Above 0.3: 0
  Above 0.1: 5
Classes: [0, 2, 56]
Class counts: [(0, 53), (2, 30), (56, 17)]

Bounding Boxes:
  Shape: torch.Size([100, 4])
  First 3 boxes: [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
  Width range: [0.00, 425.00]
  Height range: [0.00, 632.00]
  Area range: [0.00, 268600.00]
  Non-zero areas: 24 / 100

Masks:
  Type: <class 'torch.Tensor'>
  Shape: torch.Size([100, 640, 426])
  Dtype: torch.float32
  Pixels per mask (first 5): [0, 0, 0, 0, 0]
  Empty masks (0 pixels): 76 / 100
  Non-empty masks: 24
  First non-empty mask (idx 6):
    Pixels: 4871
    Score: 0.0762
    Box: [213.0, 205.0, 278.0, 325.0]