#!/bin/bash

cd "$(dirname "$0")"

declare -a models=(
    # "hyperbolic-8-16-32-resnet-20"
    "hyperbolic-8-16-32-resnet-32"
)

declare -a epsilons=(
    # "0.00314"
    # "0.00627"
    # "0.00941"
    # "0.01255"
    "0.03137"
)

declare -a datasets=(
    # "cifar10"
    "cifar100"
    # "imagenet"
)

declare -a seeds=(
    "1"
    # "2"
    # "3"
)

method=agsm
norm=-1 # 2 if l2-norm, -1 if inf-norm

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for epsilon in "${epsilons[@]}"; do
            for seed in "${seeds[@]}"; do
                echo $model
                echo $dataset
                echo $epsilon
                echo $method
                echo $norm
                echo $seed
                python -m adversarial_attacks \
                    $model $dataset \
                    -e $epsilon \
                    --batch-size 256 \
                    --norm $norm \
                    --method $method \
                    --seed $seed
            done
        done
    done
done
