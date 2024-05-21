# FTSegNet: A Novel Transformer-Based Fundus Tumor Segmentation Model Guided by Pre-trained Classification Results
Zhuo Deng, Zheng Gong, Weihao Gao, Jingyan Yang, Lei Shao, Fang Li, Wenbin Wei, **Lan Ma**

*The first two authors contribute equally to this work*

# News

* **2024.05.20** Code and models have been released. :rainbow:
* **2024.02.03** Our paper has been accepted by ISBI2024. :triangular_flag_on_post:

---

**Abstract:** Fundus tumors are the most severe retinopathies, and the deep learning segmentation model can locate the lesions and segment the lesion contour as a computer-aided diagnosis method. However, the existing algorithms for segmenting fundus tumor images have poor performance, making them unsuitable for practical clinical use. This project addresses the major issues of data and performance limitations in fundus tumor image segmentation tasks. We collect a new dataset for fundus tumor image segmentation, named FTS, which contains 254 pairs of fundus images with fundus tumor lesions and their segmentation reference images. Furthermore, a new fundus tumor segmentation network called FTSegNet is proposed in this paper. The key component in FTSegNet is the Classification Prior Block(CPB), which can provide the prior feature from classification pre-trained and guide segmentation. To better extract feature information, the Transformer and convolutional layers have been effectively combined. Qualitative and quantitative experiments are conducted on the FTS dataset to verify the effectiveness of the proposed model. We also explore the effectiveness of the CPB and different loss functions in FTSegNet. This method can provide methodological concepts for future fundus tumor segmentation tasks.

---

# 1.Create Environment:
 * Python3 (Recommend to use [Anaconda](https://www.anaconda.com/))
 * NVIDIA GPU + CUDA
 * Python packages:
   ```
   cd /FTSegNet/
   pip install -r requirements.txt
   ```

# 2.Evaluation
(1)Download the pretrained model and config from([Baidu Disk](https://pan.baidu.com/s/18Ao85fCWPNztGy4j0zkTlQ),code:fts1), and place them to `/FTSegNet/checkpoints/`.

(2)To test trained model, run
```
cd /FTSegNet/
CUDA_VISIBLE_DEVICES=0 python inference.py --model_path ./checkpoints/best_model.pth --config_path ./checkpoints/config.json --img_folder ./test_img/ --output_folder ./save_img/
```

# 3.Citation
If this repo helps you, please consider citing our work:

```
@inproceedings{FTSegNet2024,
  title={FTSegNet: A Novel Transformer-Based Fundus Tumor Segmentation Model Guided by Pre-trained Classification Results},
  author={Zhuo, Deng and Zheng, Gong and Weihao, Gao and Jingyan, Yang and Lei, Shao and Fang, Li and Wenbin, Wei and Lan, Ma},
  booktitle={2024 IEEE 20th International Symposium on Biomedical Imaging (ISBI)},
  pages={1--5},
  year={2024},
  organization={IEEE}
}
```
If you have any questions, please contact me at [malan_ailab@163.com]().
