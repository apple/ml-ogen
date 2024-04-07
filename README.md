# Overcoming the Pitfalls of Vision-Language Model Finetuning for OOD Generalization

This OGEN repository includes sample codes that can be used to finetune the CLIP model via prompt learning on various downstream datasets, with the main focus on improving the OOD GENeralization of finetuned models.

See the accompanying paper on arXiv for more details: [Overcoming the Pitfalls of Vision-Language Model Finetuning for OOD Generalization](https://arxiv.org/pdf/2401.15914.pdf)



## Getting Started

**Dependencies.** We have tested on:
- CUDA 11.8
- torch 2.0.1
- torchvision 0.15.2
- dassl 0.6.3

If PyTorch CUDA has been installed, please simply set up the environment with pip.

```shell
pip install -r requirements.txt
```

**Datasets.** To prepare all the downstream datasets (train/val/test splitting, etc), please refer to the DATASETS.md in [Link](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) and follow the instructions therein.

## Running the code

* Base-to-new generalization: training for 10 epochs
```shell
# bash scripts/ogen/base2new_train_ep10.sh <dataset_name>, <random_seed>
# dataset name: imagenet, caltech101, oxford_pets, stanford_cars, oxford_flowers, sun397, food101, fgvc_aircraft, eurosat, dtd, ucf101
# random seed: 1, 2, 3
bash scripts/ogen/base2new_train_ep10.sh caltech101 1
```

* Base-to-new generalization: evaluation after 10 epochs
```shell
# bash scripts/ogen/base2new_eval_ep10.sh <dataset_name>, <random_seed>
bash scripts/ogen/base2new_eval_ep10.sh caltech101 1
```

## Citation
```
@inproceedings{zang2023overcoming,
  title={Overcoming the Pitfalls of Vision-Language Model Finetuning for OOD Generalization},
  author={Zang, Yuhang and Goh, Hanlin and Susskind, Josh and Huang, Chen},
  booktitle={ICLR},
  year={2024}
}
```

## License

This sample code is released under the terms set forth in LICENSE.
