"""The Fast Gradient Method attack."""
import numpy as np
import torch
from models.resnets.hyperbolic import HyperbolicResNet


def angular_gradient_sign_method(
    model_fn,
    x,
    eps,
    norm=2,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    sanity_checks=False
):
    """
    PyTorch implementation of the Angular Gradient Sign Method.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # x needs to be a leaf variable, of floating point type and have requires_grad being True for
    # its grad to be computed and stored properly in a backward call
    x = x.detach().to(torch.float).requires_grad_(True)
    h = model_fn(x)
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(h, 1)

    # Compute loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(h, y)
    # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    loss.backward(retain_graph=True)

    adv_x, adv_x_ang = decompose_attack_gradients_chainrule(x, h, model_fn, x.grad, eps, norm)
    # adv_x ,adv_x_ang = decompose_attack_gradients_all_layer(x, h, model_fn, x.grad, eps, 2)

    # Add perturbation to original example to obtain adversarial example
    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
            )
        adv_x = torch.clamp(adv_x, clip_min, clip_max)
        # adv_x_rad = torch.clamp(adv_x_rad, clip_min, clip_max)
        adv_x_ang = torch.clamp(adv_x_ang, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    
    del loss
    
    return adv_x, adv_x_ang
    # return adv_x, adv_x_ang


def decompose_feature(h, h_tilde):

    delta_h = h - h_tilde                # (B, D)

    # 3) radial / angular 분해
    eps0 = 1e-12
    h_norm = h.norm(dim=-1, keepdim=True).clamp_min(eps0)
    u_h = h / h_norm                     # (B, D)
    proj = (delta_h * u_h).sum(dim=-1, keepdim=True) * u_h
    v_rad = proj                         # (B, D)
    v_ang = delta_h - proj               # (B, D)
    return v_rad, v_ang

def get_perturbed_sample(x, grad_x, eps, norm):
    if norm==2:
        adv_x = x + eps*grad_x/grad_x.norm(dim=1, keepdim=True, p=norm).clamp_min(1e-12)
    elif norm==1:
        abs_grad = torch.abs(grad_x)
        sign = torch.sign(grad_x)
        red_ind = list(range(1, len(grad_x.size())))
        abs_grad = torch.abs(grad_x)
        ori_shape = [1] * len(grad_x.size())
        ori_shape[0] = grad_x.size(0)

        max_abs_grad, _ = torch.max(abs_grad.view(grad_x.size(0), -1), 1)
        max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
        num_ties = max_mask
        for red_scalar in red_ind:
            num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
        optimal_perturbation = sign * max_mask / num_ties
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
        assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
        adv_x = x - eps * optimal_perturbation
    elif norm == np.inf:
        # Take sign of gradient
        adv_x = x + eps * torch.sign(grad_x)
    else:
        raise ValueError
    return adv_x

def decompose_attack_gradients_chainrule(x, h, model : HyperbolicResNet,
                                         grad_x, eps, norm=np.inf):
    """
    grad_x: 이미 계산된 ∇_x J (B,C,H,W)
    """

    adv_x = get_perturbed_sample(x, grad_x, eps, norm)
    
    with torch.no_grad():
        h_tilde = model(adv_x.detach())

    v_rad, v_ang = decompose_feature(h, h_tilde)

    grad_x_ang = torch.autograd.grad(
        outputs=h,
        inputs=x,
        grad_outputs=v_ang,
        retain_graph=False,
        only_inputs=True
    )[0]                                # (B, C, H, W)


    x_adv_ang = get_perturbed_sample(x, -grad_x_ang, eps, norm)
    del v_rad, grad_x_ang
    return adv_x.detach(), x_adv_ang.detach()


def angular_projected_gradient_descent(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    rand_init=True,
    rand_minmax=None,
    sanity_checks=True,
):
    """
    This class implements either the Basic Iterative Method
    (Kurakin et al. 2016) when rand_init is set to False. or the
    Madry et al. (2017) method if rand_init is set to True.
    Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
    Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param eps_iter: step size for each attack iteration
    :param nb_iter: Number of attack iterations.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
    :param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
              which the random perturbation on x was drawn. Effective only when rand_init is
              True. Default equals to eps.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm == 1:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop"
            " step for PGD when norm=1, because norm=1 FGM "
            " changes only one pixel at a time. We need "
            " to rigorously test a strong norm=1 PGD "
            "before enabling this feature."
        )
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if eps_iter < 0:
        raise ValueError(
            "eps_iter must be greater than or equal to 0, got {} instead".format(
                eps_iter
            )
        )
    if eps_iter == 0:
        return x

    assert eps_iter <= eps, (eps_iter, eps)
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # Initialize loop variables
    if rand_init:
        if rand_minmax is None:
            rand_minmax = eps
        eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
    else:
        eta = torch.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    x_adv_ang = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    i = 0
    while i < nb_iter:
        adv_x, _ = angular_gradient_sign_method(
            model_fn,
            adv_x,
            eps_iter,
            # norm,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            targeted=targeted,
        )
        _, x_adv_ang = angular_gradient_sign_method(
            model_fn,
            x_adv_ang,
            eps_iter,
            # norm,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            targeted=targeted,
        )

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
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)
        i += 1

    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        # TODO necessary to cast clip_min and clip_max to x.dtype?
        asserts.append(eps + clip_min <= clip_max)

    if sanity_checks:
        assert np.all(asserts)
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
