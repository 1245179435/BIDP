![hum](https://github.com/user-attachments/assets/fe203a1b-8051-4b6d-9dab-2a4c90f9b6f2)


# 1.BIDP 
This is DPVVNet: Dual-Process VGG-VMMamba Network for Salient Object Detectionl  code
# 2.architecture
![DP-VVNe](https://github.com/user-attachments/assets/da0cc6b6-d2e5-433a-82f7-e2c829a1ed55)

# 3.result
![十个方法对比](https://github.com/user-attachments/assets/3d88258d-1c2e-4b8f-af8b-e6b8a8e695a5)
# 4.Evaluation
For PR curve and F curve, we use the code provided by this repo: [BASNet, CVPR-2019]
For MAE, F measure, E score and S score, we use the code provided by this repo: [F3Net, AAAI-2020]
[code](https://github.com/xuebinqin/Binary-Segmentation-Evaluation-Tool)

# 5.Quantitative comparisons
![69781274a939cab817dff3f76b7152d](https://github.com/user-attachments/assets/560d6f80-a012-4114-8796-8e61ef89f713)
![0e454eb83867c2d88888fef72bd14fd](https://github.com/user-attachments/assets/4800bf07-6df6-40f4-940a-395ee6e933fa)
![PR曲线图](https://github.com/user-attachments/assets/2841a6e9-e453-4762-a39d-c1b74141ce60)

# 6.Requirements
python==3.8

torch==2.0

Torchvision

Numpy

# 7.Dataset
For all datasets, they should be organized in below's fashion:
```python
|__dataset_name

   |__Images: xxx.jpg ...
   
   |__Masks : xxx.png ...
```
download dataset
[ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
[HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html)
[DUTS-TE](http://saliencydetection.net/duts/)
[DUT-OMRON](http://saliencydetection.net/dut-omron/)
[PASCAL-S](http://cbi.gatech.edu/salobj/)
# 8.Train 
```python
pyton main.py
```




