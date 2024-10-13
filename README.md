![脑图](https://github.com/user-attachments/assets/2e54a632-85d4-4074-9fa3-2cbd98afecde)

# 1.BIDP 
This is BIDP: Brain-Inspired Dual-Process CNN-Transformer for Salient Object Detection official  code
# 2.architecture
![最终CNN(1) - 副本](https://github.com/user-attachments/assets/51f66fe0-18e2-469f-9d62-9df32e4d800b)
# 3.result
![十个方法对比](https://github.com/user-attachments/assets/3d88258d-1c2e-4b8f-af8b-e6b8a8e695a5)
# 4.Evaluation
For PR curve and F curve, we use the code provided by this repo: [BASNet, CVPR-2019]
For MAE, F measure, E score and S score, we use the code provided by this repo: [F3Net, AAAI-2020]
[code](https://github.com/xuebinqin/Binary-Segmentation-Evaluation-Tool)

# 5.Quantitative comparisons
![table1](https://github.com/user-attachments/assets/dc66b8e5-f836-46a2-9ef6-c38afa61d8de)
![table2](https://github.com/user-attachments/assets/781b6f7c-1ce5-4121-91be-cb575352c02a)
![PR曲线图](https://github.com/user-attachments/assets/2841a6e9-e453-4762-a39d-c1b74141ce60)

# 6.Requirements
python==3.8
torch==2.0

# 7.Dataset
For all datasets, they should be organized in below's fashion:
'''
|__dataset_name
   |__Images: xxx.jpg ...
   |__Masks : xxx.png ...

'''

