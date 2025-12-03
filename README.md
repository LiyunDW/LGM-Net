# <center> LGM-Net: Feature-Domain Multi-Scale Network For Image Compressive Sensing Based on Learned Gradient Mapping </center>
## <center> —— TMM 2026 —— </center>

### Authors
**Chengming Wang**<sup>1</sup>, **Yan Shen**<sup>1, 📧</sup>, **Chen Liao**<sup>1</sup>,  **Zhongli Wang**<sup>2</sup>, **Dan Li**<sup>1</sup>, **Yanbing Li**<sup>1, 📧</sup>

### Affiliations
<sup>1</sup> School of Electronic and Information Engineering, Beijing Jiaotong University, China  
<sup>2</sup> School of Automation and Intelligence, Beijing Jiaotong University, China  
<sup>📧</sup> *Corresponding author: Yan Shen, Yanbing Li*

---

## Abstract
Deep unfolding networks (DUNs), which couple traditional optimization algorithms with neural networks, have achieved remarkable success in compressive sensing. However, due to the insufficient exploitation of high-dimensional feature domain (FD) information and the loss of information during the sampling process, current DUN-based image reconstructions still suffer from significant distortions and blurring. We propose an adaptive endogenous gradient reconstruction method, LGM-Net, to address the fidelity term. First, we fully leverage the high-dimensional features derived from multiple priors and integrate them into the half quadratic splitting algorithm (HQS) iterative process, thereby transforming the conventional gradient update into the generation of endogenous gradients through adaptive endogenous gradient reconstruction (AEGR) in the FD. Second, to enhance multi-scale feature representation, we design a dual variable iterative U-Net architecture that effectively captures hierarchical structural information across FD. Finally, we introduce a customized auxiliary variable fusion (AVF) strategy during the sampling process to minimize the loss of high-frequency information caused by downsampling. Extensive experiments demonstrate that the proposed high-dimensional feature adaptive reconstruction strategy better captures global information and outperforms current state-of-the-art (SOTA) methods.

**Code Availability:** [LGM-Net](https://github.com/LiyunDW/LGM-Net)

---

## Contact Information
📧If you have any question, please email wcming@bjtu.edu.cn.

---

## Overall Architecture
![Network](./figs/model.png)
![Detail](./figs/module.png)
---
## Results
![Table_a](./figs/result_a.png)
![Table_b](./figs/result_b.png)
![Contrast](./figs/result.png)

---
## Prepare Enviroment
`conda create -n LGMNet python=3.8`

`pip install torch==1.13.0`

`pip install numpy==1.24.0 opencv-python tqdm scikit-image`
## Test
Download [official weight](https://drive.google.com/drive/folders/1wX0oh0-cnrJZZjPBNlwNwsgkR8jONftx?usp=drive_link) and put them into model folder.
We have placed the Set11 dataset in the data folder. You can test it directly by running `python test.py` without any additional steps.

If you wish to test other datasets, you can place them in the data folder and run `python test.py --test_name=["test1", "test2", "test3"]` to test all datasets at once.

## Train 
Put [CoCo2017 training set](https://cocodataset.org/#download) or others into data folder, then run `python train.py`.

## Acknowledgements
This code is built on [FSOINet](https://github.com/cwjjun/fsoinet), [OCTUF](https://github.com/songjiechong/OCTUF), [PRL](https://github.com/Guaishou74851/PRL). We thank the authors for sharing their codes.






