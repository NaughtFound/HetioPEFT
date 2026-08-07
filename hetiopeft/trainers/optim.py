from torch import nn, optim


def adam(
    model: nn.Module,
    lr: float,
    betas: tuple[float, float],
    weight_decay: float,
) -> optim.AdamW:
    return optim.AdamW(
        params=model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
