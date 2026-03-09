#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

# Modified from github.com/facebookresearch/meru

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
import torch.utils.data.dataloader
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np

from hycoclip import lorentz as L
from hycoclip.data.evaluation import CocoCaptions, Flickr30kCaptions
from hycoclip.evaluation.catalog import DatasetCatalog
from hycoclip.models import HyCoCLIP, MERU, CLIPBaseline
from hycoclip.tokenizer import Tokenizer


class ZeroShotRetrievalAttackEvaluator:
    """
    Evaluate trained models for zero-shot image and text retrieval.
    """

    def __init__(
        self,
        datasets: list[str],
        data_dir: str | Path,
        ks: list[int] = [5, 10],
        image_size: int = 224,
        eps: float = 0.00314,
        norm: int = 2,
        method: str = "agsm"
    ):
        """
        Args:
            datasets: List of dataset names to evaluate on, these names should be
                among supported datasets in `DatasetCatalog`.
            data_dir: Path to directory containing sub-directories of all datasets
                that are supported by the dataset catalog.
            ks: Top-k image/text to retrieve for calculating metrics.
            image_size: Resize images to this size for evaluation. All images
                are _squeezed_ in squares using bicubic interpolation.
        """
        self._datasets = datasets
        self._data_dir = Path(data_dir).resolve()
        self._ks = ks
        self._image_size = image_size
        self._eps = eps
        self._norm = norm
        self._method = method
        super().__init__()

    def __call__(self, model: HyCoCLIP | MERU | CLIPBaseline) -> dict[str, float]:
        model = model.eval()
        eps = self._eps
        norm = self._norm
        method = self._method

        _resize = (self._image_size, self._image_size)
        image_transform = T.Compose(
            [T.Resize(_resize, T.InterpolationMode.BICUBIC), T.ToTensor()]
        )

        # Collect results per task in this dict:
        results_dict = {}

        for dname in self._datasets:
            dataset = DatasetCatalog.build(
                dname, self._data_dir, "val", image_transform
            )
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=coco_collate)

            # Encode all images and captions.
            encoded_data = _encode_dataset_adv_i2t(data_loader, model, eps, norm, method)
            image_feats = encoded_data["image_feats"].to(model.device)
            image_feats_fgm = encoded_data["image_feats_fgm"].to(model.device)
            image_feats_agsm = encoded_data["image_feats_agsm"].to(model.device)
            text_feats = encoded_data["text_feats"].to(model.device)

            image_ids = torch.tensor(encoded_data["image_ids"])
            text_ids = torch.tensor(encoded_data["text_ids"])

            # Text-to-image retrieval: make mapping as {text_id: [sorted image_ids]}
            text_to_image_retr = {}
            text_to_image_retr_fgm = {}
            text_to_image_retr_agsm = {}

            for _ids, _queries in zip(text_ids.split(256), text_feats.split(256)):
                # Compute pairwise similarity depending on model type:
                if isinstance(model, (HyCoCLIP,MERU)):
                    scores = L.pairwise_inner(_queries, image_feats, model.curv.exp())
                    scores_fgm = L.pairwise_inner(_queries, image_feats_fgm, model.curv.exp())
                    scores_agsm = L.pairwise_inner(_queries, image_feats_agsm, model.curv.exp())
                else:
                    scores = _queries @ image_feats.T

                # Scores are "higher is better" so sort their negative, and use
                # that order to obtain image IDs per caption.
                _retrieval_order = scores.argsort(dim=1, descending=True).cpu()
                _retrieval_order_fgm = scores_fgm.argsort(dim=1, descending=True).cpu()
                _retrieval_order_agsm = scores_agsm.argsort(dim=1, descending=True).cpu()
                retrieved_image_ids = image_ids[_retrieval_order]
                retrieved_image_ids_fgm = image_ids[_retrieval_order_fgm]
                retrieved_image_ids_agsm = image_ids[_retrieval_order_agsm]

                for _id, _image_ids in zip(_ids, retrieved_image_ids):
                    text_to_image_retr[_id.item()] = _image_ids.tolist()
                for _id, _image_ids in zip(_ids, retrieved_image_ids_fgm):
                    text_to_image_retr_fgm[_id.item()] = _image_ids.tolist()
                for _id, _image_ids in zip(_ids, retrieved_image_ids_agsm):
                    text_to_image_retr_agsm[_id.item()] = _image_ids.tolist()

            # Text-to-image retrieval: make mapping as {text_id: [sorted image_ids]}
            image_to_text_retr = {}
            image_to_text_retr_fgm = {}
            image_to_text_retr_agsm = {}

            for _ids, _queries in zip(image_ids.split(256), image_feats.split(256)):
                if isinstance(model, (HyCoCLIP, MERU)):
                    scores = L.pairwise_inner(_queries, text_feats, model.curv.exp())
                else:
                    scores = _queries @ text_feats.T

                _retrieval_order = scores.argsort(dim=1, descending=True).cpu()
                retrieved_text_ids = text_ids[_retrieval_order]

                for _id, _text_ids in zip(_ids, retrieved_text_ids):
                    image_to_text_retr[_id.item()] = _text_ids.tolist()

            for _ids, _queries in zip(image_ids.split(256), image_feats_fgm.split(256)):
                if isinstance(model, (HyCoCLIP, MERU)):
                    scores = L.pairwise_inner(_queries, text_feats, model.curv.exp())
                else:
                    scores = _queries @ text_feats.T

                _retrieval_order = scores.argsort(dim=1, descending=True).cpu()
                retrieved_text_ids = text_ids[_retrieval_order]

                for _id, _text_ids in zip(_ids, retrieved_text_ids):
                    image_to_text_retr_fgm[_id.item()] = _text_ids.tolist()

            # print("ori vs fgm: ", L.pairwise_dist(image_feats, image_feats_fgm).diag().mean().data)
            # print("ori vs fgm: ", L.pairwise_dist(image_feats, image_feats_agsm).diag().mean().data)

            for _ids, _queries in zip(image_ids.split(256), image_feats_agsm.split(256)):
                if isinstance(model, (HyCoCLIP, MERU)):
                    scores = L.pairwise_inner(_queries, text_feats, model.curv.exp())
                else:
                    scores = _queries @ text_feats.T

                _retrieval_order = scores.argsort(dim=1, descending=True).cpu()
                retrieved_text_ids = text_ids[_retrieval_order]

                for _id, _text_ids in zip(_ids, retrieved_text_ids):
                    image_to_text_retr_agsm[_id.item()] = _text_ids.tolist()

            # Compute text-to-image and image-to-text recall@K for both datasets.
            for _k in self._ks:
                results_dict[f"{dname}_t2i_r{_k}"] = _compute_recall(
                    text_to_image_retr, encoded_data["text_to_image_gt"], _k
                )
                results_dict[f"{dname}_t2i_r{_k}_fgm"] = _compute_recall(
                    text_to_image_retr_fgm, encoded_data["text_to_image_gt"], _k
                )
                results_dict[f"{dname}_t2i_r{_k}_agsm"] = _compute_recall(
                    text_to_image_retr_agsm, encoded_data["text_to_image_gt"], _k
                )

            for _k in self._ks:
                results_dict[f"{dname}_i2t_r{_k}"] = _compute_recall(
                    image_to_text_retr, encoded_data["image_to_text_gt"], _k
                )
                results_dict[f"{dname}_i2t_r{_k}_fgm"] = _compute_recall(
                    image_to_text_retr_fgm, encoded_data["image_to_text_gt"], _k
                )
                results_dict[f"{dname}_i2t_r{_k}_agsm"] = _compute_recall(
                    image_to_text_retr_agsm, encoded_data["image_to_text_gt"], _k
                )

        return results_dict

def _encode_dataset_adv_i2t(
    data_loader: CocoCaptions | Flickr30kCaptions,
    model: HyCoCLIP | MERU | CLIPBaseline,
    eps: float,
    norm: int,
    method: str
):
    """
    Extract image-text features and their instance IDs using a given dataset
    (COCO or Flickr30k) and a given model (MERU or CLIP).
    """

    encoded_data = {
        "image_ids": [],
        "text_ids": [],
        "image_feats": [],
        "image_feats_fgm": [],
        "image_feats_agsm": [],
        "text_feats": [],
        # Dict mapping as {image_id: {matching_text_ids} } and vice-versa.
        "image_to_text_gt": defaultdict(set),
        "text_to_image_gt": defaultdict(set),
    }

    tokenizer = Tokenizer()
    criterion = torch.nn.CrossEntropyLoss()
    print(f"--------------\neps: {eps}\nnorm: {norm}\nmethod: {method}\n--------------")

    for inst in tqdm(data_loader, desc="Extracting image-text features"):
    # for inst in data_loader:
        # Add entries to ground-truth dict.
        image_id = inst["image_id"]
        
        for img_id, cap_ids in zip(image_id, inst["caption_ids"]):
            if isinstance(cap_ids, torch.Tensor):
                cap_ids = cap_ids.tolist()  # ex: [381347, 381368, ...]

            # image → text GT
            encoded_data["image_to_text_gt"][img_id].update(cap_ids)

            # text → image GT
            for cid in cap_ids:
                encoded_data["text_to_image_gt"][cid].add(img_id)


        inst["caption_ids"] = torch.cat(inst["caption_ids"], dim=0)

        x = inst["image"].to(model.device).requires_grad_(True)

        captions = []
        captions_for_adv = []
        for c in inst["captions"]: # inst["captions"] : 5, inst["captions"][i] : bs
            captions += c
            captions_for_adv.append(c[0])
        # print(captions) ; exit()
        with torch.no_grad():
            caption_tokens = tokenizer(captions)
            caption_feats = model.encode_text(caption_tokens, project=True)
        caption_adv_tokens = tokenizer(captions_for_adv)
        caption_adv_feat = model.encode_text(caption_adv_tokens, project=True)

        if method=="agsm":
            adv_x, x_adv_ang = fast_gradient_method(caption_adv_feat, model, x, eps, norm)
        elif method=="apgd":
            adv_x, x_adv_ang = projected_gradient_descent(model,x,eps,eps/10,20,caption_adv_feat, norm)
        
        with torch.no_grad():
            image_feats = model.encode_image(
                x, project=True
            )
            image_feats_adv = model.encode_image(
                adv_x, project=True
            )
            image_feats_adv_ang = model.encode_image(
                x_adv_ang, project=True
            )
        # Add current entries to extracted features and IDs.
        encoded_data["image_ids"].extend(inst["image_id"])
        encoded_data["image_feats"].append(image_feats.detach().cpu())
        encoded_data["image_feats_fgm"].append(image_feats_adv.detach().cpu())
        encoded_data["image_feats_agsm"].append(image_feats_adv_ang.detach().cpu())
        encoded_data["text_ids"].extend(inst["caption_ids"])
        encoded_data["text_feats"].append(caption_feats.detach().cpu())

    # shape: (dataset_size, model.embed_dim), (dataset_size, model.embed_dim)
    encoded_data["image_feats"] = torch.cat(encoded_data["image_feats"], dim=0)
    encoded_data["image_feats_fgm"] = torch.cat(encoded_data["image_feats_fgm"], dim=0)
    encoded_data["image_feats_agsm"] = torch.cat(encoded_data["image_feats_agsm"], dim=0)
    encoded_data["text_feats"] = torch.cat(encoded_data["text_feats"], dim=0)

    return encoded_data


def get_perturbed_sample(x, grad_x, eps, norm):
    if norm==2:
        adv_x = x + eps*grad_x/grad_x.norm(dim=1, keepdim=True, p=norm).clamp_min(1e-12)
    elif norm == np.inf:
        # Take sign of gradient
        adv_x = x + eps * torch.sign(grad_x)
    else:
        raise ValueError
    return adv_x


def fast_gradient_method(caption_adv_feat, model, x, eps, norm):
    x = x.detach().requires_grad_(True)
    image_feats = model.encode_image(
        x, project=True
    ) # bs x 512
    sim_matrix = L.pairwise_inner(image_feats, caption_adv_feat, model.curv.exp())
    # targets = torch.arange(image_feats.size(0), device=image_feats.device)
    targets = torch.argmax(sim_matrix, dim=1)
    
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(sim_matrix, targets)
    model.zero_grad()
    loss.backward(retain_graph=True)

    grad_x = x.grad

    adv_x = get_perturbed_sample(x, grad_x, eps, norm)
    # adv_x = x + eps * grad_x/grad_x.norm(dim=1, keepdim=True).clamp_min(1e-12)

    image_feats_adv = model.encode_image(
        adv_x, project=False
    )
    h = L.log_map0(image_feats, curv=model.curv.exp())
    h_adv = image_feats_adv
    _, v_ang = _decompose_feature(h, h_adv)

    grad_x_ang = torch.autograd.grad(
        outputs=h,
        inputs=x,
        grad_outputs=v_ang,
        retain_graph=False,
        only_inputs=True
    )[0]                                # (B, C, H, W)
    x_adv_ang = get_perturbed_sample(x, -grad_x_ang, eps, norm)
    return adv_x.detach(), x_adv_ang.detach()


def projected_gradient_descent(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    caption_adv_feat,
    norm,
):
    
    # Initialize loop variables
    eta = torch.zeros_like(x).uniform_(-eps, eps)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    x_adv_ang = x + eta

    i = 0
    while i < nb_iter:
        adv_x, _ = fast_gradient_method(caption_adv_feat, model_fn, adv_x, eps_iter, norm)
        _, x_adv_ang = fast_gradient_method(caption_adv_feat, model_fn, x_adv_ang, eps_iter, norm)

        with torch.no_grad():
            # Clipping perturbation eta to norm norm ball
            eta = adv_x - x
            eta = clip_eta(eta, norm, eps)
            adv_x = x + eta

            eta = x_adv_ang - x
            eta = clip_eta(eta, norm, eps)
            x_adv_ang = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        i += 1

    return adv_x.detach(), x_adv_ang.detach()


def clip_eta(eta, norm, eps):
    """
    PyTorch implementation of the clip_eta in utils_tf.

    :param eta: Tensor
    :param norm: np.inf, 1, or 2
    :param eps: float
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")

    avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
    # reduc_ind = list(range(1, len(eta.size())))
    if norm == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("L1 clip is not implemented.")
            norm = torch.max(
                avoid_zero_div, torch.sum(torch.abs(eta), dim=1, keepdim=True)
            )
        elif norm == 2:
            norm = torch.sqrt(
                torch.max(
                    avoid_zero_div, torch.sum(eta ** 2, dim=1, keepdim=True)
                )
            )
        factor = torch.min(
            torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm
        )
        eta *= factor
    return eta

def _decompose_feature(h, h_tilde):

    delta_h = h - h_tilde                # (B, D)

    # 3) radial / angular decomposition
    eps0 = 1e-12
    h_norm = h.norm(dim=-1, keepdim=True).clamp_min(eps0)
    u_h = h / h_norm                     # (B, D)
    proj = (delta_h * u_h).sum(dim=-1, keepdim=True) * u_h
    v_rad = proj                         # (B, D)
    v_ang = delta_h - proj               # (B, D)
    return v_rad, v_ang



def _compute_recall(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    K: int,
):
    """
    Compute recall @ K for COCO and Flickr30K image/text retrieval.

    Args:
        predictions: Dict with integer keys representing image (or text) IDs, and
            values being a ranked list of retrieved text (or image) IDs.
        ground_truth: Dict with integer keys representing image (or text) IDs
            (same as `predictions`) and values being a list of integer IDs
            of the paired text/images.
        K: Measure recall among Top-K retrievals.

    Returns:
        Single float value giving the average recall@k across all ground-truth.
    """

    num_correct_retrievals = 0.0
    for query_id, paired_ids in ground_truth.items():
        predictions_id = predictions.get(query_id, [])

        if set(predictions_id[:K]) & paired_ids:
            num_correct_retrievals += 1.0

    return 100.0 * num_correct_retrievals / len(ground_truth)

def coco_collate(batch: list[dict]):

    image_ids   = [item["image_id"] for item in batch]
    caption_ids = [torch.tensor(item["caption_ids"]) for item in batch]   # list of 1D tensors
    images      = torch.stack([item["image"] for item in batch], dim=0)   # (B, C, H, W)
    captions    = [item["captions"] for item in batch]                   # list of list[str]
    return {
        "image_id": image_ids,
        "caption_ids": caption_ids,
        "image": images,
        "captions": captions,
    }
