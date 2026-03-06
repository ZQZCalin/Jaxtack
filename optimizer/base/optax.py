# Wrapper for optax-implemented optimizers.
# https://optax.readthedocs.io/en/latest/api/optimizers.html

from __future__ import annotations

from typing import Any, Callable
from jax.typing import ArrayLike
from optimizer.config import BaseOptimizerConfig

import optax
from optimizer.registry import OPTIMIZER_REGISTRY

# Type alias for mask parameters (tree or callable).
MaskOrFn = Any | Callable[[optax.Params], Any] | None


# -----------------------------------------------------------------------------
# AdaBelief
# -----------------------------------------------------------------------------
class AdaBeliefConfig(BaseOptimizerConfig):
    """Optax AdaBelief.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adabelief
    """
    name: str = "adabelief"
    b1: ArrayLike = 0.9
    b2: ArrayLike = 0.999
    eps: ArrayLike = 1e-16
    eps_root: ArrayLike = 1e-16
    nesterov: bool = False


@OPTIMIZER_REGISTRY.register("adabelief", AdaBeliefConfig)
def build_adabelief(
    config: AdaBeliefConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.adabelief(
        learning_rate=learning_rate,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        eps_root=config.eps_root,
        nesterov=config.nesterov,
    )


# -----------------------------------------------------------------------------
# AdaDelta
# -----------------------------------------------------------------------------
class AdaDeltaConfig(BaseOptimizerConfig):
    """Optax AdaDelta.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adadelta
    """
    name: str = "adadelta"
    rho: ArrayLike = 0.9
    eps: ArrayLike = 1e-6
    weight_decay: ArrayLike = 0.0
    weight_decay_mask: MaskOrFn = None


@OPTIMIZER_REGISTRY.register("adadelta", AdaDeltaConfig)
def build_adadelta(
    config: AdaDeltaConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.adadelta(
        learning_rate=learning_rate,
        rho=config.rho,
        eps=config.eps,
        weight_decay=config.weight_decay,
        weight_decay_mask=config.weight_decay_mask,
    )


# -----------------------------------------------------------------------------
# Adan
# -----------------------------------------------------------------------------
class AdanConfig(BaseOptimizerConfig):
    """Optax Adan (ADAptive Nesterov momentum).
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adan
    """
    name: str = "adan"
    b1: ArrayLike = 0.98
    b2: ArrayLike = 0.92
    b3: ArrayLike = 0.99
    eps: ArrayLike = 1e-8
    eps_root: ArrayLike = 1e-8
    weight_decay: ArrayLike = 0.0
    mask: MaskOrFn = None


@OPTIMIZER_REGISTRY.register("adan", AdanConfig)
def build_adan(
    config: AdanConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.adan(
        learning_rate=learning_rate,
        b1=config.b1,
        b2=config.b2,
        b3=config.b3,
        eps=config.eps,
        eps_root=config.eps_root,
        weight_decay=config.weight_decay,
        mask=config.mask,
    )


# -----------------------------------------------------------------------------
# AdaGrad
# -----------------------------------------------------------------------------
class AdaGradConfig(BaseOptimizerConfig):
    """Optax AdaGrad.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adagrad
    """
    name: str = "adagrad"
    initial_accumulator_value: ArrayLike = 0.1
    eps: ArrayLike = 1e-7


@OPTIMIZER_REGISTRY.register("adagrad", AdaGradConfig)
def build_adagrad(
    config: AdaGradConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.adagrad(
        learning_rate=learning_rate,
        initial_accumulator_value=config.initial_accumulator_value,
        eps=config.eps,
    )


# -----------------------------------------------------------------------------
# AdaFactor
# -----------------------------------------------------------------------------
class AdaFactorConfig(BaseOptimizerConfig):
    """Optax AdaFactor.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adafactor
    """
    name: str = "adafactor"
    min_dim_size_to_factor: int = 128
    decay_rate: ArrayLike = 0.8
    decay_offset: ArrayLike = 0
    multiply_by_parameter_scale: bool = True
    clipping_threshold: ArrayLike | None = 1.0
    momentum: ArrayLike | None = None
    dtype_momentum: Any = None  # jnp.float32
    weight_decay_rate: ArrayLike | None = None
    eps: ArrayLike = 1e-30
    factored: bool = True
    weight_decay_mask: MaskOrFn = None


@OPTIMIZER_REGISTRY.register("adafactor", AdaFactorConfig)
def build_adafactor(
    config: AdaFactorConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.adafactor(
        learning_rate=learning_rate,
        min_dim_size_to_factor=config.min_dim_size_to_factor,
        decay_rate=config.decay_rate,
        decay_offset=config.decay_offset,
        multiply_by_parameter_scale=config.multiply_by_parameter_scale,
        clipping_threshold=config.clipping_threshold,
        momentum=config.momentum,
        dtype_momentum=config.dtype_momentum,
        weight_decay_rate=config.weight_decay_rate,
        eps=config.eps,
        factored=config.factored,
        weight_decay_mask=config.weight_decay_mask,
    )


# -----------------------------------------------------------------------------
# Adam
# -----------------------------------------------------------------------------
class AdamConfig(BaseOptimizerConfig):
    """Optax Adam.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adam
    """
    name: str = "adam"
    b1: ArrayLike = 0.9
    b2: ArrayLike = 0.999
    eps: ArrayLike = 1e-8
    eps_root: ArrayLike = 0.0
    mu_dtype: Any | None = None
    nesterov: bool = False


@OPTIMIZER_REGISTRY.register("adam", AdamConfig)
def build_adam(
    config: AdamConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.adam(
        learning_rate=learning_rate,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        eps_root=config.eps_root,
        mu_dtype=config.mu_dtype,
        nesterov=config.nesterov,
    )


# -----------------------------------------------------------------------------
# Adamax
# -----------------------------------------------------------------------------
class AdamaxConfig(BaseOptimizerConfig):
    """Optax Adamax (Adam with infinity norm).
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adamax
    """
    name: str = "adamax"
    b1: ArrayLike = 0.9
    b2: ArrayLike = 0.999
    eps: ArrayLike = 1e-8


@OPTIMIZER_REGISTRY.register("adamax", AdamaxConfig)
def build_adamax(
    config: AdamaxConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.adamax(
        learning_rate=learning_rate,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
    )


# -----------------------------------------------------------------------------
# AdamaxW
# -----------------------------------------------------------------------------
class AdamaxWConfig(BaseOptimizerConfig):
    """Optax AdamaxW (Adamax with weight decay).
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adamaxw
    """
    name: str = "adamaxw"
    b1: ArrayLike = 0.9
    b2: ArrayLike = 0.999
    eps: ArrayLike = 1e-8
    weight_decay: ArrayLike = 0.0001
    mask: MaskOrFn = None


@OPTIMIZER_REGISTRY.register("adamaxw", AdamaxWConfig)
def build_adamaxw(
    config: AdamaxWConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.adamaxw(
        learning_rate=learning_rate,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        weight_decay=config.weight_decay,
        mask=config.mask,
    )


# -----------------------------------------------------------------------------
# AdamW
# -----------------------------------------------------------------------------
class AdamWConfig(BaseOptimizerConfig):
    """Optax AdamW (Adam with weight decay).
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adamw
    """
    name: str = "adamw"
    b1: ArrayLike = 0.9
    b2: ArrayLike = 0.999
    eps: ArrayLike = 1e-08
    eps_root: ArrayLike = 0.0
    mu_dtype: Any | None = None
    weight_decay: ArrayLike = 0.0001
    mask: MaskOrFn = None
    nesterov: bool = False


@OPTIMIZER_REGISTRY.register("adamw", AdamWConfig)
def build_adamw(
    config: AdamWConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.adamw(
        learning_rate=learning_rate,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        eps_root=config.eps_root,
        mu_dtype=config.mu_dtype,
        weight_decay=config.weight_decay,
        mask=config.mask,
        nesterov=config.nesterov,
    )


# -----------------------------------------------------------------------------
# AMSGrad
# -----------------------------------------------------------------------------
class AMSGradConfig(BaseOptimizerConfig):
    """Optax AMSGrad.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.amsgrad
    """
    name: str = "amsgrad"
    b1: ArrayLike = 0.9
    b2: ArrayLike = 0.999
    eps: ArrayLike = 1e-8
    eps_root: ArrayLike = 0.0
    mu_dtype: Any | None = None
    bias_correction_mu: bool = True
    bias_correction_nu: bool = True


@OPTIMIZER_REGISTRY.register("amsgrad", AMSGradConfig)
def build_amsgrad(
    config: AMSGradConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.amsgrad(
        learning_rate=learning_rate,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        eps_root=config.eps_root,
        mu_dtype=config.mu_dtype,
        bias_correction_mu=config.bias_correction_mu,
        bias_correction_nu=config.bias_correction_nu,
    )


# -----------------------------------------------------------------------------
# Fromage
# -----------------------------------------------------------------------------
class FromageConfig(BaseOptimizerConfig):
    """Optax Fromage (Frobenius matched gradient descent).
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.fromage
    """
    name: str = "fromage"
    min_norm: ArrayLike = 1e-6


@OPTIMIZER_REGISTRY.register("fromage", FromageConfig)
def build_fromage(
    config: FromageConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.fromage(
        learning_rate=learning_rate,
        min_norm=config.min_norm,
    )


# -----------------------------------------------------------------------------
# LAMB
# -----------------------------------------------------------------------------
class LambConfig(BaseOptimizerConfig):
    """Optax LAMB.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.lamb
    """
    name: str = "lamb"
    b1: ArrayLike = 0.9
    b2: ArrayLike = 0.999
    eps: ArrayLike = 1e-6
    eps_root: ArrayLike = 0.0
    weight_decay: ArrayLike = 0.0
    mask: MaskOrFn = None


@OPTIMIZER_REGISTRY.register("lamb", LambConfig)
def build_lamb(
    config: LambConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.lamb(
        learning_rate=learning_rate,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        eps_root=config.eps_root,
        weight_decay=config.weight_decay,
        mask=config.mask,
    )


# -----------------------------------------------------------------------------
# LARS
# -----------------------------------------------------------------------------
class LarsConfig(BaseOptimizerConfig):
    """Optax LARS.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.lars
    """
    name: str = "lars"
    weight_decay: ArrayLike = 0.0
    weight_decay_mask: MaskOrFn = True
    trust_coefficient: ArrayLike = 0.001
    eps: ArrayLike = 0.0
    trust_ratio_mask: MaskOrFn = True
    momentum: ArrayLike = 0.9
    nesterov: bool = False


@OPTIMIZER_REGISTRY.register("lars", LarsConfig)
def build_lars(
    config: LarsConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.lars(
        learning_rate=learning_rate,
        weight_decay=config.weight_decay,
        weight_decay_mask=config.weight_decay_mask,
        trust_coefficient=config.trust_coefficient,
        eps=config.eps,
        trust_ratio_mask=config.trust_ratio_mask,
        momentum=config.momentum,
        nesterov=config.nesterov,
    )


# -----------------------------------------------------------------------------
# L-BFGS
# -----------------------------------------------------------------------------
class LBFGSConfig(BaseOptimizerConfig):
    """Optax L-BFGS.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.lbfgs
    """
    name: str = "lbfgs"
    memory_size: int = 10
    scale_init_precond: bool = True


@OPTIMIZER_REGISTRY.register("lbfgs", LBFGSConfig)
def build_lbfgs(
    config: LBFGSConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.lbfgs(
        learning_rate=learning_rate,
        memory_size=config.memory_size,
        scale_init_precond=config.scale_init_precond,
    )


# -----------------------------------------------------------------------------
# Lion
# -----------------------------------------------------------------------------
class LionConfig(BaseOptimizerConfig):
    """Optax Lion.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.lion
    """
    name: str = "lion"
    b1: ArrayLike = 0.9
    b2: ArrayLike = 0.99
    mu_dtype: Any | None = None
    weight_decay: ArrayLike = 0.001
    mask: MaskOrFn = None


@OPTIMIZER_REGISTRY.register("lion", LionConfig)
def build_lion(
    config: LionConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.lion(
        learning_rate=learning_rate,
        b1=config.b1,
        b2=config.b2,
        mu_dtype=config.mu_dtype,
        weight_decay=config.weight_decay,
        mask=config.mask,
    )


# -----------------------------------------------------------------------------
# NAdam
# -----------------------------------------------------------------------------
class NAdamConfig(BaseOptimizerConfig):
    """Optax NAdam (Adam with Nesterov momentum).
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.nadam
    """
    name: str = "nadam"
    b1: ArrayLike = 0.9
    b2: ArrayLike = 0.999
    eps: ArrayLike = 1e-8
    eps_root: ArrayLike = 0.0
    mu_dtype: Any | None = None
    nesterov: bool = True


@OPTIMIZER_REGISTRY.register("nadam", NAdamConfig)
def build_nadam(
    config: NAdamConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.nadam(
        learning_rate=learning_rate,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        eps_root=config.eps_root,
        mu_dtype=config.mu_dtype,
        nesterov=config.nesterov,
    )


# -----------------------------------------------------------------------------
# NAdamW
# -----------------------------------------------------------------------------
class NAdamWConfig(BaseOptimizerConfig):
    """Optax NAdamW (NAdam with weight decay).
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.nadamw
    """
    name: str = "nadamw"
    b1: ArrayLike = 0.9
    b2: ArrayLike = 0.999
    eps: ArrayLike = 1e-8
    eps_root: ArrayLike = 0.0
    mu_dtype: Any | None = None
    weight_decay: ArrayLike = 0.0001
    mask: MaskOrFn = None
    nesterov: bool = True


@OPTIMIZER_REGISTRY.register("nadamw", NAdamWConfig)
def build_nadamw(
    config: NAdamWConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.nadamw(
        learning_rate=learning_rate,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        eps_root=config.eps_root,
        mu_dtype=config.mu_dtype,
        weight_decay=config.weight_decay,
        mask=config.mask,
        nesterov=config.nesterov,
    )


# -----------------------------------------------------------------------------
# Noisy SGD
# -----------------------------------------------------------------------------
class NoisySGDConfig(BaseOptimizerConfig):
    """Optax Noisy SGD.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.noisy_sgd
    """
    name: str = "noisy_sgd"
    eta: ArrayLike = 0.01
    gamma: ArrayLike = 0.55
    key: ArrayLike | None = None
    seed: int | None = None


@OPTIMIZER_REGISTRY.register("noisy_sgd", NoisySGDConfig)
def build_noisy_sgd(
    config: NoisySGDConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.noisy_sgd(
        learning_rate=learning_rate,
        eta=config.eta,
        gamma=config.gamma,
        key=config.key,
        seed=config.seed,
    )


# -----------------------------------------------------------------------------
# NovoGrad
# -----------------------------------------------------------------------------
class NovoGradConfig(BaseOptimizerConfig):
    """Optax NovoGrad.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.novograd
    """
    name: str = "novograd"
    b1: ArrayLike = 0.9
    b2: ArrayLike = 0.25
    eps: ArrayLike = 1e-6
    eps_root: ArrayLike = 0.0
    weight_decay: ArrayLike = 0.0


@OPTIMIZER_REGISTRY.register("novograd", NovoGradConfig)
def build_novograd(
    config: NovoGradConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.novograd(
        learning_rate=learning_rate,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        eps_root=config.eps_root,
        weight_decay=config.weight_decay,
    )


# -----------------------------------------------------------------------------
# Optimistic Gradient Descent
# -----------------------------------------------------------------------------
class OptimisticGradientDescentConfig(BaseOptimizerConfig):
    """Optax Optimistic Gradient Descent.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.optimistic_gradient_descent
    """
    name: str = "optimistic_gradient_descent"
    alpha: ArrayLike = 1.0
    beta: ArrayLike = 1.0


@OPTIMIZER_REGISTRY.register("optimistic_gradient_descent", OptimisticGradientDescentConfig)
def build_optimistic_gradient_descent(
    config: OptimisticGradientDescentConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.optimistic_gradient_descent(
        learning_rate=learning_rate,
        alpha=config.alpha,
        beta=config.beta,
    )


# -----------------------------------------------------------------------------
# Optimistic Adam v2
# -----------------------------------------------------------------------------
class OptimisticAdamV2Config(BaseOptimizerConfig):
    """Optax Optimistic Adam v2.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.optimistic_adam_v2
    """
    name: str = "optimistic_adam_v2"
    alpha: ArrayLike = 1.0
    beta: ArrayLike = 1.0
    b1: ArrayLike = 0.9
    b2: ArrayLike = 0.999
    eps: ArrayLike = 1e-8
    eps_root: ArrayLike = 0.0
    mu_dtype: Any | None = None
    nesterov: bool = True


@OPTIMIZER_REGISTRY.register("optimistic_adam_v2", OptimisticAdamV2Config)
def build_optimistic_adam_v2(
    config: OptimisticAdamV2Config,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.optimistic_adam_v2(
        learning_rate=learning_rate,
        alpha=config.alpha,
        beta=config.beta,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        eps_root=config.eps_root,
        mu_dtype=config.mu_dtype,
        nesterov=config.nesterov,
    )


# -----------------------------------------------------------------------------
# Polyak SGD
# -----------------------------------------------------------------------------
class PolyakSGDConfig(BaseOptimizerConfig):
    """Optax SGD with Polyak step-size.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.polyak_sgd
    """
    name: str = "polyak_sgd"
    max_learning_rate: ArrayLike = 1.0
    scaling: ArrayLike = 1.0
    f_min: ArrayLike = 0.0
    eps: ArrayLike = 0.0
    variant: str = "sps"


@OPTIMIZER_REGISTRY.register("polyak_sgd", PolyakSGDConfig)
def build_polyak_sgd(
    config: PolyakSGDConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    # polyak_sgd uses max_learning_rate as main LR; we pass schedule as scaling
    # or use config.max_learning_rate. Here we use learning_rate as max_learning_rate.
    return optax.polyak_sgd(
        max_learning_rate=learning_rate,
        scaling=config.scaling,
        f_min=config.f_min,
        eps=config.eps,
        variant=config.variant,
    )


# -----------------------------------------------------------------------------
# RAdam
# -----------------------------------------------------------------------------
class RAdamConfig(BaseOptimizerConfig):
    """Optax RAdam (Rectified Adam).
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.radam
    """
    name: str = "radam"
    b1: ArrayLike = 0.9
    b2: ArrayLike = 0.999
    eps: ArrayLike = 1e-8
    eps_root: ArrayLike = 0.0
    threshold: ArrayLike = 5.0
    nesterov: bool = False


@OPTIMIZER_REGISTRY.register("radam", RAdamConfig)
def build_radam(
    config: RAdamConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.radam(
        learning_rate=learning_rate,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        eps_root=config.eps_root,
        threshold=config.threshold,
        nesterov=config.nesterov,
    )


# -----------------------------------------------------------------------------
# RMSProp
# -----------------------------------------------------------------------------
class RMSPropConfig(BaseOptimizerConfig):
    """Optax RMSProp.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.rmsprop
    """
    name: str = "rmsprop"
    decay: ArrayLike = 0.9
    eps: ArrayLike = 1e-8
    initial_scale: ArrayLike = 0.0
    eps_in_sqrt: bool = True
    centered: bool = False
    momentum: ArrayLike | None = None
    nesterov: bool = False
    bias_correction: bool = False


@OPTIMIZER_REGISTRY.register("rmsprop", RMSPropConfig)
def build_rmsprop(
    config: RMSPropConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.rmsprop(
        learning_rate=learning_rate,
        decay=config.decay,
        eps=config.eps,
        initial_scale=config.initial_scale,
        eps_in_sqrt=config.eps_in_sqrt,
        centered=config.centered,
        momentum=config.momentum,
        nesterov=config.nesterov,
        bias_correction=config.bias_correction,
    )


# -----------------------------------------------------------------------------
# RProp
# -----------------------------------------------------------------------------
class RPropConfig(BaseOptimizerConfig):
    """Optax RProp.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.rprop
    """
    name: str = "rprop"
    learning_rate: ArrayLike = 0.003  # initial step size
    eta_minus: ArrayLike = 0.5
    eta_plus: ArrayLike = 1.2
    min_step_size: ArrayLike = 1e-6
    max_step_size: ArrayLike = 50.0


@OPTIMIZER_REGISTRY.register("rprop", RPropConfig)
def build_rprop(
    config: RPropConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    # RProp uses a scalar initial step size; use builder LR if scalar else config
    lr = (
        learning_rate
        if isinstance(learning_rate, (int, float))
        else config.learning_rate
    )
    return optax.rprop(
        learning_rate=lr,
        eta_minus=config.eta_minus,
        eta_plus=config.eta_plus,
        min_step_size=config.min_step_size,
        max_step_size=config.max_step_size,
    )


# -----------------------------------------------------------------------------
# SGD
# -----------------------------------------------------------------------------
class SGDConfig(BaseOptimizerConfig):
    """Optax SGD.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.sgd
    """
    name: str = "sgd"
    momentum: ArrayLike | None = None
    nesterov: bool = False
    accumulator_dtype: Any | None = None


@OPTIMIZER_REGISTRY.register("sgd", SGDConfig)
def build_sgd(
    config: SGDConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.sgd(
        learning_rate=learning_rate,
        momentum=config.momentum,
        nesterov=config.nesterov,
        accumulator_dtype=config.accumulator_dtype,
    )


# -----------------------------------------------------------------------------
# SignSGD
# -----------------------------------------------------------------------------
class SignSGDConfig(BaseOptimizerConfig):
    """Optax SignSGD.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.sign_sgd
    """
    name: str = "sign_sgd"


@OPTIMIZER_REGISTRY.register("sign_sgd", SignSGDConfig)
def build_sign_sgd(
    config: SignSGDConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.sign_sgd(learning_rate=learning_rate)


# -----------------------------------------------------------------------------
# Signum
# -----------------------------------------------------------------------------
class SignumConfig(BaseOptimizerConfig):
    """Optax Signum.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.signum
    """
    name: str = "signum"
    beta: ArrayLike = 0.9
    accumulator_dtype: Any | None = None


@OPTIMIZER_REGISTRY.register("signum", SignumConfig)
def build_signum(
    config: SignumConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.signum(
        learning_rate=learning_rate,
        beta=config.beta,
        accumulator_dtype=config.accumulator_dtype,
    )


# -----------------------------------------------------------------------------
# SM3
# -----------------------------------------------------------------------------
class SM3Config(BaseOptimizerConfig):
    """Optax SM3.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.sm3
    """
    name: str = "sm3"
    momentum: ArrayLike = 0.9


@OPTIMIZER_REGISTRY.register("sm3", SM3Config)
def build_sm3(
    config: SM3Config,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.sm3(
        learning_rate=learning_rate,
        momentum=config.momentum,
    )


# -----------------------------------------------------------------------------
# Yogi
# -----------------------------------------------------------------------------
class YogiConfig(BaseOptimizerConfig):
    """Optax Yogi.
    https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.yogi
    """
    name: str = "yogi"
    b1: ArrayLike = 0.9
    b2: ArrayLike = 0.999
    eps: ArrayLike = 0.001


@OPTIMIZER_REGISTRY.register("yogi", YogiConfig)
def build_yogi(
    config: YogiConfig,
    learning_rate: optax.ScalarOrSchedule,
) -> optax.GradientTransformation:
    return optax.yogi(
        learning_rate=learning_rate,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
    )
