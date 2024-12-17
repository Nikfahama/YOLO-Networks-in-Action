# Applied Statistics Final Project - YOLO Networks in Action

## Introduction and Project Outline

In this project we will be exploring the process of retraining the some of the most popular state of the art computer vision models out there - more specifically, 
multiple samples from the YOLO (You Only Look Once) family of architectures.  
  
The YOLO series relies on several gnerations / implementations of different architectures to achieve object / bounding box detections on classes which they were trained on. YOLO was originally made
for real time applications, however for the scope of our project we will focus strictly on single frame performance and testing. We have included a snippet of output from running inference of a YOLOv8m
model we retrained as to provide an example of what the model is capable of doing. In this specific example the model found one plane with a confidence of 90%.

<img src="images/example_YOLOv8m.jpg" alt="Testing" width="500" />  
  
### Now for the outline of our project  
In our case specifically we will be retraining 4 variants of YOLO on the same dataset to examine each model's respective performance and characteristics. The motivation for this project
is to exemplify how one might implement these models for their specific needs in the real world - for example an intelligence agency whos job it is to examine satellite photos, which can use YOLO
in order to sweep though a large amount of photos hastily while being able to retrain the model on new targets that they might need to identify (a new type of plane, a dock / harbour or anything else for that matter).  
  
The variants of YOLO that we will be retraining are:  

- YOLOv5n
- YOLOv5m
- YOLOv8n
- YOLOv8m

### What are the differences between these variants?  
  
**As a general rule of thumb** YOLO models come in multiple *sizes* - N, S, M, L, X. The meanings of each can be summarised using the following table:  
  
  | Model        | Size       | Speed         | Accuracy       |
|--------------|------------|---------------|----------------|
| **YOLO-N**   | Very Small | Fastest       | Lowest Accuracy|
| **YOLO-S**   | Small      | Very Fast     | Lower Accuracy |
| **YOLO-M**   | Medium     | Balanced      | Medium Accuracy|
| **YOLO-L**   | Large      | Slower        | High Accuracy  |
| **YOLO-X**   | Extra Large| Slowest       | Highest Accuracy|
  
**There are architectural differences between YOLOv5 and YOLOv8**  

## Below is a summary of the architectures of the YOLO variants we selected to examine:  

<div style="display: flex; gap: 20px;">

<div>

### YOLOv5n

| **Index** | **From** | **#** | **Params** | **Module**                              | **Arguments**                  |
|-----------|----------|-------|------------|----------------------------------------|--------------------------------|
| 0         | -1       | 1     | 1,760      | ultralytics.nn.modules.conv.Conv       | [3, 16, 6, 2, 2]               |
| 1         | -1       | 1     | 4,672      | ultralytics.nn.modules.conv.Conv       | [16, 32, 3, 2]                 |
| 2         | -1       | 1     | 4,800      | ultralytics.nn.modules.block.C3        | [32, 32, 1]                    |
| 3         | -1       | 1     | 18,560     | ultralytics.nn.modules.conv.Conv       | [32, 64, 3, 2]                 |
| 4         | -1       | 2     | 29,184     | ultralytics.nn.modules.block.C3        | [64, 64, 2]                    |
| 5         | -1       | 1     | 73,984     | ultralytics.nn.modules.conv.Conv       | [64, 128, 3, 2]                |
| 6         | -1       | 3     | 156,928    | ultralytics.nn.modules.block.C3        | [128, 128, 3]                  |
| 7         | -1       | 1     | 295,424    | ultralytics.nn.modules.conv.Conv       | [128, 256, 3, 2]               |
| 8         | -1       | 1     | 296,448    | ultralytics.nn.modules.block.C3        | [256, 256, 1]                  |
| 9         | -1       | 1     | 164,608    | ultralytics.nn.modules.block.SPPF      | [256, 256, 5]                  |
| 10        | -1       | 1     | 33,024     | ultralytics.nn.modules.conv.Conv       | [256, 128, 1, 1]               |
| 11        | -1       | 1     | 0          | torch.nn.modules.upsampling.Upsample   | [None, 2, 'nearest']           |
| 12        | [-1, 6]  | 1     | 0          | ultralytics.nn.modules.conv.Concat     | [1]                            |
| 13        | -1       | 1     | 90,880     | ultralytics.nn.modules.block.C3        | [256, 128, 1, False]           |
| 14        | -1       | 1     | 8,320      | ultralytics.nn.modules.conv.Conv       | [128, 64, 1, 1]                |
| 15        | -1       | 1     | 0          | torch.nn.modules.upsampling.Upsample   | [None, 2, 'nearest']           |
| 16        | [-1, 4]  | 1     | 0          | ultralytics.nn.modules.conv.Concat     | [1]                            |
| 17        | -1       | 1     | 22,912     | ultralytics.nn.modules.block.C3        | [128, 64, 1, False]            |
| 18        | -1       | 1     | 36,992     | ultralytics.nn.modules.conv.Conv       | [64, 64, 3, 2]                 |
| 19        | [-1, 14] | 1     | 0          | ultralytics.nn.modules.conv.Concat     | [1]                            |
| 20        | -1       | 1     | 74,496     | ultralytics.nn.modules.block.C3        | [128, 128, 1, False]           |
| 21        | -1       | 1     | 147,712    | ultralytics.nn.modules.conv.Conv       | [128, 128, 3, 2]               |
| 22        | [-1, 10] | 1     | 0          | ultralytics.nn.modules.conv.Concat     | [1]                            |
| 23        | -1       | 1     | 296,448    | ultralytics.nn.modules.block.C3        | [256, 256, 1, False]           |
| 24        | [17, 20, 23] | 1 | 754,237    | ultralytics.nn.modules.head.Detect     | [15, [64, 128, 256]]           |


### Key Points:

- **Total Layers**: 262  
- **Total Trainable Parameters**: 2.51M  

</div>

<div>

### YOLOv5m

| **Index** | **From**         | **n** | **Params**  | **Module**                                  | **Arguments**                      |
|-----------|------------------|-------|-------------|--------------------------------------------|-----------------------------------|
| 0         | -1               | 1     | 5280        | ultralytics.nn.modules.conv.Conv            | [3, 48, 6, 2, 2]                  |
| 1         | -1               | 1     | 41664       | ultralytics.nn.modules.conv.Conv            | [48, 96, 3, 2]                    |
| 2         | -1               | 2     | 65280       | ultralytics.nn.modules.block.C3             | [96, 96, 2]                       |
| 3         | -1               | 1     | 166272      | ultralytics.nn.modules.conv.Conv            | [96, 192, 3, 2]                   |
| 4         | -1               | 4     | 444672      | ultralytics.nn.modules.block.C3             | [192, 192, 4]                     |
| 5         | -1               | 1     | 664320      | ultralytics.nn.modules.conv.Conv            | [192, 384, 3, 2]                  |
| 6         | -1               | 6     | 2512896     | ultralytics.nn.modules.block.C3             | [384, 384, 6]                     |
| 7         | -1               | 1     | 2655744     | ultralytics.nn.modules.conv.Conv            | [384, 768, 3, 2]                  |
| 8         | -1               | 2     | 4134912     | ultralytics.nn.modules.block.C3             | [768, 768, 2]                     |
| 9         | -1               | 1     | 1476864     | ultralytics.nn.modules.block.SPPF           | [768, 768, 5]                     |
| 10        | -1               | 1     | 295680      | ultralytics.nn.modules.conv.Conv            | [768, 384, 1, 1]                  |
| 11        | -1               | 1     | 0           | torch.nn.modules.upsampling.Upsample        | [None, 2, 'nearest']              |
| 12        | [-1, 6]          | 1     | 0           | ultralytics.nn.modules.conv.Concat          | [1]                               |
| 13        | -1               | 2     | 1182720     | ultralytics.nn.modules.block.C3             | [768, 384, 2, False]              |
| 14        | -1               | 1     | 74112       | ultralytics.nn.modules.conv.Conv            | [384, 192, 1, 1]                  |
| 15        | -1               | 1     | 0           | torch.nn.modules.upsampling.Upsample        | [None, 2, 'nearest']              |
| 16        | [-1, 4]          | 1     | 0           | ultralytics.nn.modules.conv.Concat          | [1]                               |
| 17        | -1               | 2     | 296448      | ultralytics.nn.modules.block.C3             | [384, 192, 2, False]              |
| 18        | -1               | 1     | 332160      | ultralytics.nn.modules.conv.Conv            | [192, 192, 3, 2]                  |
| 19        | [-1, 14]         | 1     | 0           | ultralytics.nn.modules.conv.Concat          | [1]                               |
| 20        | -1               | 2     | 1035264     | ultralytics.nn.modules.block.C3             | [384, 384, 2, False]              |
| 21        | -1               | 1     | 1327872     | ultralytics.nn.modules.conv.Conv            | [384, 384, 3, 2]                  |
| 22        | [-1, 10]         | 1     | 0           | ultralytics.nn.modules.conv.Concat          | [1]                               |
| 23        | -1               | 2     | 4134912     | ultralytics.nn.modules.block.C3             | [768, 768, 2, False]              |
| 24        | [17, 20, 23]     | 1     | 4226749     | ultralytics.nn.modules.head.Detect          | [15, [192, 384, 768]]             |


### Key Points:

- **Total Layers**: 339  
- **Total Trainable Parameters**: 25.07M  

</div>

</div>
