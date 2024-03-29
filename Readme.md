# [AAAI2022] Detecting Human-Object Interactions with Object-Guided Cross-Modal Calibrated Semantics 

<div align="center">
  <img src=".github/OCN_pipeline.png" width="600px" />
  <p>Overall pipeline of OCN.</p>
</div>
 
Paper Link: [[AAAI official paper]](https://ojs.aaai.org/index.php/AAAI/article/view/20229)
[[arXiv]](https://arxiv.org/abs/2202.00259)

[![GitHub Stars](https://img.shields.io/github/stars/JacobYuan7/OCN-HOI-Benchmark?style=social)](https://github.com/JacobYuan7/OCN-HOI-Benchmark)
[![GitHub Forks](https://img.shields.io/github/forks/JacobYuan7/OCN-HOI-Benchmark)](https://github.com/JacobYuan7/OCN-HOI-Benchmark)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FJacobYuan7%2FOCN-HOI-Benchmark%2F&count_bg=%235FC1D7&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)
![visitors](https://visitor-badge.glitch.me/badge?page_id=JacobYuan7/OCN-HOI-Benchmark)

💥**News**! The follow-up work [**RLIPv2: Fast Scaling of Relational Language-Image Pre-training**](https://arxiv.org/abs/2308.09351) is accepted to **ICCV 2023**. Its code have been released in [RLIPv2 repo](https://github.com/JacobYuan7/RLIPv2). 

💥**News**! The follow-up work [**RLIP: Relational Language-Image Pre-training**](https://arxiv.org/abs/2209.01814) is accepted to **NeurIPS 2022** as a **Spotlight** paper (Top 5%) and also available online! [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2209.01814) Hope you will enjoy reading it.

If you find our work or the codebase inspiring and useful to your research, please cite
```bibtex
@inproceedings{Yuan2022OCN,
  title={Detecting Human-Object Interactions with Object-Guided Cross-Modal Calibrated Semantics},
  author={Hangjie Yuan and Mang Wang and Dong Ni and Liangpeng Xu},
  booktitle={AAAI},
  year={2022}
}

@inproceedings{Yuan2022RLIP,
  title={RLIP: Relational Language-Image Pre-training for Human-Object Interaction Detection},
  author={Yuan, Hangjie and Jiang, Jianwen and Albanie, Samuel and Feng, Tao and Huang, Ziyuan and Ni, Dong and Tang, Mingqian},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}

@inproceedings{Yuan2023RLIPv2,
  title={RLIPv2: Fast Scaling of Relational Language-Image Pre-training},
  author={Yuan, Hangjie and Zhang, Shiwei and Wang, Xiang and Albanie, Samuel and Pan, Yining and Feng, Tao and Jiang, Jianwen and Ni, Dong and Zhang, Yingya and Zhao, Deli},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```
## Dataset preparation
### 1. HICO-DET
HICO-DET dataset can be downloaded [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk). After finishing downloading, unpack the tarball (`hico_20160224_det.tar.gz`) to the `data` directory.

Instead of using the original annotations files, we use the annotation files provided by the PPDM authors. The annotation files can be downloaded from [here](https://drive.google.com/open?id=1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R). The downloaded annotation files have to be placed as follows.
```
qpic
 |─ data
 │   └─ hico_20160224_det
 |       |─ annotations
 |       |   |─ trainval_hico.json
 |       |   |─ test_hico.json
 |       |   └─ corre_hico.npy
 :       :
```

### 2. V-COCO
First clone the repository of V-COCO from [here](https://github.com/s-gupta/v-coco), and then follow the instruction to generate the file `instances_vcoco_all_2014.json`. Next, download the prior file `prior.pickle` from [here](https://drive.google.com/drive/folders/10uuzvMUCVVv95-xAZg5KS94QXm7QXZW4). Place the files and make directories as follows.
```
qpic
 |─ data
 │   └─ v-coco
 |       |─ data
 |       |   |─ instances_vcoco_all_2014.json
 |       |   :
 |       |─ prior.pickle
 |       |─ images
 |       |   |─ train2014
 |       |   |   |─ COCO_train2014_000000000009.jpg
 |       |   |   :
 |       |   └─ val2014
 |       |       |─ COCO_val2014_000000000042.jpg
 |       |       :
 |       |─ annotations
 :       :
```
For our implementation, the annotation file have to be converted to the HOIA format. The conversion can be conducted as follows.
```
PYTHONPATH=data/v-coco \
        python convert_vcoco_annotations.py \
        --load_path data/v-coco/data \
        --prior_path data/v-coco/prior.pickle \
        --save_path data/v-coco/annotations
```
Note that only Python2 can be used for this conversion because `vsrl_utils.py` in the v-coco repository shows a error with Python3.

V-COCO annotations with the HOIA format, `corre_vcoco.npy`, `test_vcoco.json`, and `trainval_vcoco.json` will be generated to `annotations` directory.

## Dependencies and Training
To simplify the steps, we combine the installation of externel dependencies and training into one '.sh' file. You can directly run the codes after rightly preparing the dataset.
```
# Training on HICO-DET
bash train_hico.sh
# Training on V-COCO
bash train_vcoco.sh
```
Note that you can refer to the publicly available [codebase](https://github.com/hitachi-rd-cv/qpic) for the preparation of two datasets.


## Pre-trained parameters
OCN uses COCO pretrained models for fair comparisons with previous methods. The pretrained models can be downloaded from [DETR](https://github.com/facebookresearch/detr) repository. 

For HICO-DET, you can convert the pre-trained parameters with the following command.
```
python convert_parameters.py \
        --load_path /PATH/TO/PRETRAIN \
        --save_path /PATH/TO/SAVE
```
For V-COCO, you can convert the pre-trained parameters with the following command.
```
python convert_parameters.py \
        --load_path /PATH/TO/PRETRAIN \
        --save_path /PATH/TO/SAVE \
        --dataset vcoco \
```



## Evaluation
The mAP on HICO-DET under the Full set, Rare set and Non-Rare Set will be reported during the training process. Or you can evaluate the performance using commands below:
```
python main.py \
    --pretrained /PATH/TO/PRETRAINED_MODEL \
    --output_dir /PATH/TO/OUTPUT \
    --hoi \
    --dataset_file hico \
    --hoi_path /PATH/TO/data/hico_20160224_det \
    --num_obj_classes 80 \
    --num_verb_classes 117 \
    --backbone resnet101 \
    --num_workers 4 \
    --batch_size 4 \
    --exponential_hyper 1 \
    --exponential_loss \
    --semantic_similar_coef 1 \
    --verb_loss_type focal \
    --semantic_similar \
    --OCN \
    --eval \
```

The results for the official evaluation of V-COCO must be obtained by the generated pickle file of detection results.
```
python generate_vcoco_official.py \
        --param_path /PATH/TO/CHECKPOINT \
        --save_path /PATH/TO/SAVE/vcoco.pickle \
        --hoi_path /PATH/TO/VCOCO/data/v-coco \
        --batch_size 4 \
        --OCN \
```
Then you should run following codes after modifying the path to get the final performance:
```
python datasets/vsrl_eval.py
```

## Results
We present the results and links for downloading corresponding parameters and logs below. Results are evaluated in Known Object setting. We evaluate the model from the last epoch of training. (The checkpoints can produce higher results than what are reported in the paper.) Results and parameters on HICO-DET can be found in the table below: 
| Model | Backbone | Rare | None-Rare | Full | Download |
| ---------- | :-----------:  | :-----------:  | :-----------: | :-----------: | :-----------: |
| OCN | ResNet-50 | 25.56 | 32.92 | 31.23 | [link](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/EcaO1pep2XtKoG-8U9NFvfkBNpG5n34Tb_ccxeMbOdo6Sg?e=hcbspJ) |
| OCN | ResNet-101 | 26.24 | 33.27 | 31.65 | [link](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/EaDjNk2OZpFCpNRYIzmBZI4BMbO7NglqiWoPqfO9hcKzOg?e=Crsuw7) |

Results and parameters on V-COCO can be found in the table below:
| Model | Backbone | $AP_{role}^{1}$ | $AP_{role}^{2}$ | Download |
| ---------- | :-----------:  | :-----------:  | :-----------: | :-----------: |
| OCN | ResNet-50 | 64.2 | 66.3 | [link](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/EejYy8FsT4tPpsjjOvn2ZDsBckBu2C2g7eA0javN055MAQ?e=vQLTmm) |
| OCN | ResNet-101 | 65.3 | 67.1 | [link](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/EZiXop8K4VJGu_dB9eTMIVABiCM6K4p_r0I3saUdQgtuLg?e=Pl5Yjh) |
