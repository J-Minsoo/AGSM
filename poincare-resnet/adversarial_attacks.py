import argparse
import os

import numpy as np
import torch
import torch.nn as nn
# from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from attack_utils.fast_gradient_method import angular_gradient_sign_method, angular_projected_gradient_descent
from timm import utils
from tqdm import tqdm
import random

from cifar10.dataloader import Cifar10DataLoaderFactory
from cifar100.dataloader import Cifar100DataLoaderFactory
from imagenet.dataloader import TinyImageNetDataLoaderFactory
from models.resnets import parse_model_from_name

parser = argparse.ArgumentParser(description="PyTorch adversarial attack evaluation")

parser.add_argument("model", type=str, help="Model name")
parser.add_argument("dataset", type=str, help="Dataset name")
parser.add_argument("-e", "--epsilon", type=float)
parser.add_argument("-b", "--batch-size", type=int, default=128)
parser.add_argument("--norm", type=int, default=2)
parser.add_argument("--method", type=str, default="fgm")
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--seed", type=int, default=3)


def create_metrics_dict():
    return {
        "losses": utils.AverageMeter(),
        "top1": utils.AverageMeter(),
        "top5": utils.AverageMeter(),
    }

def set_seed(seed: int = 3):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def update_metrics_dict(metrics, input, loss, acc1, acc5):
    metrics["losses"].update(loss.data.item(), input.size(0))
    metrics["top1"].update(acc1.item(), input.size(0))
    metrics["top5"].update(acc5.item(), input.size(0))
    return metrics


def main() -> dict:
    args = parser.parse_args()
    set_seed(int(args.seed))

    if args.dataset == "cifar10":
        _, test_loader = Cifar10DataLoaderFactory.create_train_loaders(
            batch_size=args.batch_size
        )
        classes = 10
    elif args.dataset == "cifar100":
        _, test_loader = Cifar100DataLoaderFactory.create_train_loaders(
            batch_size=args.batch_size
        )
        classes = 100
    elif args.dataset == "imagenet":
        dataset_factory = TinyImageNetDataLoaderFactory("/path/to/root/directory/containing/tiny-imagenet-200")
        classes = 200
        train_loader, test_loader = dataset_factory.create_train_loaders(
            batch_size=args.batch_size
        )


    model = parse_model_from_name(model_name=args.model, classes=classes).cuda()
    print(test_loader.sampler)

    if args.resume:
        weights_path = args.resume
    else:
        weights_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "weights",
            args.dataset,
            args.model,
            f"{args.model}_weights.pth",
        )
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    metrics = {attack: create_metrics_dict() for attack in ["clean", "fgm", "agsm"]}

    for input, target in tqdm(test_loader, total=len(test_loader)):
    # for input, target in test_loader:
        input, target = input.cuda(), target.cuda()
        if args.method == "agsm":
            adv_x, input_agsms = angular_gradient_sign_method(
                model_fn=model, x=input, eps=args.epsilon, norm=args.norm if args.norm!= -1 else torch.inf
            )

        elif args.method == "apgd":
            adv_x, input_agsms = angular_projected_gradient_descent(
                model_fn=model, x=input, eps=args.epsilon, eps_iter=args.epsilon/4, nb_iter=20, norm=args.norm if args.norm!= -1 else torch.inf
            )

        input = input.detach()
        adv_x = adv_x.detach()
        input_agsms = input_agsms.detach()
        with torch.no_grad():
            output = model(input.detach())
            output_fgm = model(adv_x)
            output_agsm = model(input_agsms)
            # print((input-input_agsm).norm(dim=1))


        loss = loss_fn(output, target)
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        metrics["clean"] = update_metrics_dict(
            metrics["clean"], input, loss, acc1, acc5
        )

        loss_fgm = loss_fn(output_fgm, target)
        acc1_fgm, acc5_fgm = utils.accuracy(output_fgm, target, topk=(1, 5))
        metrics["fgm"] = update_metrics_dict(
            metrics["fgm"], output_fgm.detach(), loss_fgm, acc1_fgm, acc5_fgm
        )
        loss_fgm = loss_fn(output_agsm, target)
        acc1_fgm, acc5_fgm = utils.accuracy(output_agsm, target, topk=(1, 5))
        metrics[f"agsm"] = update_metrics_dict(
            metrics[f"agsm"], output_agsm.detach(), loss_fgm, acc1_fgm, acc5_fgm
        )

        del loss, loss_fgm, output_fgm, output_agsm, output, target
        torch.cuda.empty_cache()

    for attack in metrics.keys():
        print(
            f"Metrics for {attack}:  "
            f"Loss: {metrics[attack]['losses'].avg:>7.4f}  "
            f"Acc@1: {metrics[attack]['top1'].avg:>7.4f}  "
            f"Acc@5: {metrics[attack]['top5'].avg:>7.4f}"
        )

if __name__ == "__main__":
    main()
