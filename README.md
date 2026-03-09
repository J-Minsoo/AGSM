# Angular Gradient Sign Method (AGSM)

PyTorch code for Angular Gradient Sign Method experiments on hyperbolic image classification and vision-language retrieval. This repository currently combines two subprojects:

- `poincare-resnet`: hyperbolic image classification and adversarial evaluation
- `hycoclip`: hyperbolic vision-language retrieval evaluation

The main AGSM implementation for classification lives in `poincare-resnet/attack_utils/fast_gradient_method.py`, and the retrieval attack evaluation lives in `hycoclip/hycoclip/evaluation/attack.py`.

This repository builds on [HyCoCLIP](https://github.com/PalAvik/hycoclip), [Poincare ResNet](https://github.com/maxvanspengler/poincare-resnet), and the [cleverhans](https://github.com/cleverhans-lab/cleverhans) package. The AGSM baseline code in this repository was implemented with reference to `cleverhans`, and the overall codebase also references the upstream `hycoclip` and `poincare-resnet` repositories.

## Environment

Use a Conda environment named `agsm`.

```bash
conda create -n agsm python=3.12 -y
conda activate agsm
```

Install the dependencies for each subproject separately. Install the PyTorch and torchvision build appropriate for your CUDA setup before running the experiments.

```bash
pip install -r poincare-resnet/requirements.txt
pip install -r hycoclip/requirements.txt
pip install -e ./hycoclip
```

## Data and checkpoints

### `poincare-resnet`

- Update `poincare-resnet/config.ini` with the dataset root paths used for CIFAR-10, CIFAR-100, Tiny-ImageNet, Places365, SVHN, and Textures.
- The default `poincare-resnet/adversarial_attacks.sh` configuration runs AGSM on CIFAR-100 with the pretrained hyperbolic ResNet-32 checkpoint under `poincare-resnet/weights/`.

### `hycoclip`

- Place the retrieval checkpoint at `hycoclip/checkpoints/hycoclip_vit_s.pth`.
- Update `data_dir` in `hycoclip/configs/eval_zero_shot_retrieval_attack.py` so it points to the directory containing the evaluation datasets.
- The retrieval attack configuration currently expects [COCO VAL 2017](https://cocodataset.org/#download) and [Flickr30k test](https://huggingface.co/datasets/nlphuji/flickr30k).

## Running experiments

### Hyperbolic image classification attack

From the repository root:

```bash
bash poincare-resnet/adversarial_attacks.sh
```

### Zero-shot retrieval attack

From the `hycoclip` directory:

```bash
python scripts/evaluate.py --config configs/eval_zero_shot_retrieval_attack.py \
    --checkpoint-path checkpoints/hycoclip_vit_s.pth \
    --train-config configs/train_hycoclip_vit_s.py --seed 1
```

## Repository layout

```text
AGSM/
├── poincare-resnet/
│   ├── adversarial_attacks.sh
│   ├── adversarial_attacks.py
│   └── attack_utils/fast_gradient_method.py
└── hycoclip/
    ├── scripts/evaluate.py
    ├── configs/eval_zero_shot_retrieval_attack.py
    └── hycoclip/evaluation/attack.py
```

## Citation

If you use this repository in academic work, please cite the [AGSM](https://arxiv.org/abs/2511.12985) paper.
