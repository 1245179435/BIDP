

![脑图](https://github.com/user-attachments/assets/e1f87201-b3a6-4daa-8667-f33a676e2305)

# 1.BIDP 
This is BIDP: Brain-Inspired Dual-Process CNN-Transformer for
Salient Object Detectionl  code
# 2.architecture

![figurez](https://github.com/user-attachments/assets/e980faf2-16bd-429e-a0dd-0cb4aa0d9152)


# 3.result
![十个方法对比](https://github.com/user-attachments/assets/3d88258d-1c2e-4b8f-af8b-e6b8a8e695a5)
# 4.Evaluation
For PR curve and F curve, we use the code provided by this repo: [BASNet, CVPR-2019]
For MAE, F measure, E score and S score, we use the code provided by this repo: [F3Net, AAAI-2020]
[code](https://github.com/xuebinqin/Binary-Segmentation-Evaluation-Tool)

# 5.Quantitative comparisons
![1744348456426](https://github.com/user-attachments/assets/004f0a31-3c46-4f4c-bd91-aa6541655c7f)

<img width="775" alt="PR" src="https://github.com/user-attachments/assets/012f36aa-d2f1-42b5-be68-30c2e7fe354d" />


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




