# PAKDD2020
### 比赛链接：https://tianchi.aliyun.com/competition/entrance/231775/introduction?spm=5176.12281915.0.0.26bf1020wwszHt

# 团队名称：橘猫
## 复赛：Rank9

## 代码说明：
### 特征思路
对具有相同的raw和norm特征进行除法操作，并滑窗统计mean，std等特征。
### 模型训练
构建label利用距离真实损坏的天数，用lgb进行回归预测。
### 执行顺序
1. gen_train_test_data.py
2. lgb_model.py
3. main.py
