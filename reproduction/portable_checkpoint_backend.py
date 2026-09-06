"""Small PyTorch DCP backend used by the VeRL checkpoint overlay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful


class _AppState(Stateful):
    """DCP Stateful adapter that can rebuild an empty optimizer state on load."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        *,
        include_model: bool,
        include_optimizer: bool,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.include_model = include_model
        self.include_optimizer = include_optimizer

    def state_dict(self) -> dict[str, Any]:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
            get_state_dict,
        )

        options = StateDictOptions(full_state_dict=False, cpu_offload=True)
        if self.include_optimizer:
            if self.optimizer is None:
                raise ValueError(
                    "Optimizer state was requested but no optimizer was provided"
                )
            model_state, optimizer_state = get_state_dict(
                self.model, self.optimizer, options=options
            )
            state: dict[str, Any] = {"optimizer": optimizer_state}
            if self.include_model:
                state["model"] = model_state
            return state
        if self.include_model:
            return {"model": get_model_state_dict(self.model, options=options)}
        raise ValueError(
            "Portable checkpoint requested without model or optimizer state"
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
            set_state_dict,
        )

        options = StateDictOptions(full_state_dict=False, cpu_offload=True)
        if self.include_optimizer:
            if self.optimizer is None:
                raise ValueError(
                    "Optimizer state was requested but no optimizer was provided"
                )
            set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=state_dict.get("model", {}),
                optim_state_dict=state_dict["optimizer"],
                options=options,
            )
        elif self.include_model:
            set_model_state_dict(self.model, state_dict["model"], options=options)


def _contents(*, model: bool, optimizer: bool) -> list[str]:
    contents = []
    if model:
        contents.append("model")
    if optimizer:
        contents.append("optimizer")
    if not contents:
        raise ValueError(
            "Portable checkpoint requested without model or optimizer state"
        )
    return contents


def _prime_empty_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Allocate Adam-style tensor slots so DCP has in-place load destinations.

    DCP restores tensors into pre-allocated storage. A freshly constructed AdamW
    optimizer has no moment tensors, so perform a zero-learning-rate step before
    asking DCP for the load skeleton. The real model and optimizer states replace
    these placeholders immediately afterward.
    """

    if optimizer.state:
        return
    saved_group_values = [
        (group.get("lr"), group.get("weight_decay")) for group in optimizer.param_groups
    ]
    saved_grads: list[tuple[torch.nn.Parameter, torch.Tensor | None]] = []
    try:
        for group in optimizer.param_groups:
            group["lr"] = 0.0
            if "weight_decay" in group:
                group["weight_decay"] = 0.0
            for parameter in group["params"]:
                saved_grads.append((parameter, parameter.grad))
                parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
    finally:
        for parameter, grad in saved_grads:
            parameter.grad = grad
        for group, (learning_rate, weight_decay) in zip(
            optimizer.param_groups, saved_group_values, strict=True
        ):
            group["lr"] = learning_rate
            if weight_decay is not None:
                group["weight_decay"] = weight_decay


def save_model_optimizer(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    save_model: bool,
    save_optimizer: bool,
) -> list[str]:
    import torch.distributed.checkpoint as dcp

    contents = _contents(model=save_model, optimizer=save_optimizer)
    app_state = _AppState(
        model,
        optimizer,
        include_model=save_model,
        include_optimizer=save_optimizer,
    )
    dcp.save(
        {"app": app_state},
        checkpoint_id=str(checkpoint_dir),
        no_dist=not torch.distributed.is_initialized(),
    )
    return contents


def load_model_optimizer(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    load_model: bool,
    load_optimizer: bool,
) -> list[str]:
    import torch.distributed.checkpoint as dcp

    contents = _contents(model=load_model, optimizer=load_optimizer)
    if load_optimizer:
        if optimizer is None:
            raise ValueError(
                "Optimizer state was requested but no optimizer was provided"
            )
        _prime_empty_optimizer_state(optimizer)
    app_state = _AppState(
        model,
        optimizer,
        include_model=load_model,
        include_optimizer=load_optimizer,
    )
    dcp.load(
        {"app": app_state},
        checkpoint_id=str(checkpoint_dir),
        no_dist=not torch.distributed.is_initialized(),
    )
    return contents
