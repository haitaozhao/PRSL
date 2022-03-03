# read the data

```python
import csv
import numpy as np

with open('score.csv', newline='') as f:
    reader = csv.reader(f)
    s = list(reader)
tmp = s[0]
score = np.array([float(item) for item in tmp])

with open('label.csv', newline='') as f:
    reader = csv.reader(f)
    l = list(reader)
tmp = l[0]
label = np.array([float(item) for item in tmp])
```



## 问题：

1. 针对以上的数据，将label为1的作为P类，把label为0的作为N类，阈值设为0.05时，计算混淆矩阵，计算TPR，FPR，Precision，Recall，F1-score，Accuracy
2. 绘制ROC曲线，计算AUC

