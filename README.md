# 清华大学大数据分析课程：期末项目 - 沃尔玛销售预测

## 项目背景
本项目是清华大学大数据分析课程的期末作业，目标是应用数据分析技术对沃尔玛的销售数据进行预测。项目基于Kaggle的[M5 Forecasting - Accuracy竞赛](https://www.kaggle.com/competitions/m5-forecasting-accuracy)。

## 环境配置指南
请执行以下命令以安装必要的Python依赖包：
```
pip install -r requirements.txt
```

## 数据集安置
确保将 `m5-forecasting-accuracy` 数据集下载后放置于项目根目录。

## 深度学习模型训练及预测
执行以下脚本以进行模型训练和销售预测：
```
bash ./scripts/m5.sh
```
预测结果将保存在 `submission.csv` 文件中。

## LGBM模型训练及预测
Coming soon...

## 提交指南
完成预测后，将结果文件提交至Kaggle平台，以便与其他参赛者的成果进行比较。