###########################################################################################
# Training script
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed
import torch_geometric
from torch.nn.parallel import DistributedDataParallel
from torch.optim import LBFGS
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_ema import ExponentialMovingAverage
from torchmetrics import Metric

from mace.cli.visualise_train import TrainingPlotter
from mace.tools.checkpoint import CheckpointHandler, CheckpointState
from mace.tools.torch_tools import to_numpy
from mace.tools.utils import (MetricsLogger, compute_mae, compute_q95,
                              compute_rel_mae, compute_rel_rmse, compute_rmse)


@dataclasses.dataclass
class SWAContainer:
    model: AveragedModel
    scheduler: SWALR
    start: int
    loss_fn: torch.nn.Module


def valid_err_log(
    valid_loss,
    eval_metrics,
    logger,
    log_errors,
    epoch=None,
    valid_loader_name="Default",
):
    eval_metrics["mode"] = "eval"
    eval_metrics["epoch"] = epoch
    eval_metrics["head"] = valid_loader_name
    logger.log(eval_metrics)
    if epoch is None:
        inintial_phrase = "Initial"
    else:
        inintial_phrase = f"Epoch {epoch}"
    if log_errors == "PerAtomRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A"
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_stress"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_stress = eval_metrics["rmse_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_stress={error_stress:8.2f} meV / A^3",
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_virials = eval_metrics["rmse_virials_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_virials_per_atom={error_virials:8.2f} meV",
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_stress_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_stress = eval_metrics["mae_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_stress={error_stress:8.2f} meV / A^3"
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_virials = eval_metrics["mae_virials"] * 1e3
        logging.info(
            f"{inintial_phrase}: loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_virials={error_virials:8.2f} meV"
        )
    elif log_errors == "TotalRMSE":
        error_e = eval_metrics["rmse_e"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "PerAtomMAE":
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "TotalMAE":
        error_e = eval_metrics["mae_e"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "DipoleRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_MU_per_atom={error_mu:8.2f} mDebye",
        )
    elif log_errors == "EnergyDipoleRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_Mu_per_atom={error_mu:8.2f} mDebye",
        )


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    start_epoch: int,
    max_num_epochs: int,
    patience: int,
    checkpoint_handler: CheckpointHandler,
    logger: MetricsLogger,
    eval_interval: int,
    output_args: Dict[str, bool],
    device: torch.device,
    log_errors: str,
    swa: Optional[SWAContainer] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    max_grad_norm: Optional[float] = 10.0,
    log_wandb: bool = False,
    distributed: bool = False,
    save_all_checkpoints: bool = False,
    plotter: TrainingPlotter = None,
    distributed_model: Optional[DistributedDataParallel] = None,
    train_sampler: Optional[DistributedSampler] = None,
    rank: Optional[int] = 0,
):
    lowest_loss = np.inf
    valid_loss = np.inf
    patience_counter = 0
    swa_start = True
    keep_last = False
    if log_wandb:
        import wandb

    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")

    logging.info("")
    logging.info("===========Precision Test (precision)===========")
    logging.info("Started training, reporting errors on validation set")
    logging.info("Loss metrics on validation set")
    epoch = start_epoch

    # log validation loss before _any_ training
    for valid_loader_name, valid_loader in valid_loaders.items():
        valid_loss_head, eval_metrics = evaluate(
            model=model,
            loss_fn=loss_fn,
            data_loader=valid_loader,
            output_args=output_args,
            device=device,
        )
        valid_err_log(
            valid_loss_head, eval_metrics, logger, log_errors, None, valid_loader_name
        )
    valid_loss = valid_loss_head  # consider only the last head for the checkpoint

    logging.info("Precision test complete")


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    ema: Optional[ExponentialMovingAverage],
    logger: MetricsLogger,
    device: torch.device,
    distributed: bool,
    distributed_model: Optional[DistributedDataParallel] = None,
    rank: Optional[int] = 0,
) -> None:
    model_to_train = model if distributed_model is None else distributed_model

    if isinstance(optimizer, LBFGS):
        _, opt_metrics = take_step_lbfgs(
            model=model_to_train,
            loss_fn=loss_fn,
            data_loader=data_loader,
            optimizer=optimizer,
            ema=ema,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            device=device,
            distributed=distributed,
            rank=rank,
        )
        opt_metrics["mode"] = "opt"
        opt_metrics["epoch"] = epoch
        if rank == 0:
            logger.log(opt_metrics)
    else:
        for batch in data_loader:
            _, opt_metrics = take_step(
                model=model_to_train,
                loss_fn=loss_fn,
                batch=batch,
                optimizer=optimizer,
                ema=ema,
                output_args=output_args,
                max_grad_norm=max_grad_norm,
                device=device,
            )
            opt_metrics["mode"] = "opt"
            opt_metrics["epoch"] = epoch
            if rank == 0:
                logger.log(opt_metrics)


def take_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    batch = batch.to(device)
    batch_dict = batch.to_dict()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch_dict,
            training=True,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        loss = loss_fn(pred=output, ref=batch)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        return loss

    loss = closure()
    optimizer.step()

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


def take_step_lbfgs(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
    distributed: bool,
    rank: int,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    logging.debug(
        f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )

    total_sample_count = 0
    for batch in data_loader:
        total_sample_count += batch.num_graphs

    if distributed:
        global_sample_count = torch.tensor(total_sample_count, device=device)
        torch.distributed.all_reduce(
            global_sample_count, op=torch.distributed.ReduceOp.SUM
        )
        total_sample_count = global_sample_count.item()

    signal = torch.zeros(1, device=device) if distributed else None

    def closure():
        if distributed:
            if rank == 0:
                signal.fill_(1)
                torch.distributed.broadcast(signal, src=0)

            for param in model.parameters():
                torch.distributed.broadcast(param.data, src=0)

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)

        # Process each batch and then collect the results we pass to the optimizer
        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            output = model(
                batch_dict,
                training=True,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
            batch_loss = loss_fn(pred=output, ref=batch)
            batch_loss = batch_loss * (batch.num_graphs / total_sample_count)

            batch_loss.backward()
            total_loss += batch_loss

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        if distributed:
            torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        return total_loss

    if distributed:
        if rank == 0:
            loss = optimizer.step(closure)
            signal.fill_(0)
            torch.distributed.broadcast(signal, src=0)
        else:
            while True:
                # Other ranks wait for signals from rank 0
                torch.distributed.broadcast(signal, src=0)
                if signal.item() == 0:
                    break
                if signal.item() == 1:
                    loss = closure()

        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)
    else:
        loss = optimizer.step(closure)

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    output_args: Dict[str, bool],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    for param in model.parameters():
        param.requires_grad = False

    metrics = MACELoss(loss_fn=loss_fn).to(device)

    start_time = time.time()
    for batch in data_loader:
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        output = model(
            batch_dict,
            training=False,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        avg_loss, aux = metrics(batch, output)

    avg_loss, aux = metrics.compute()
    aux["time"] = time.time() - start_time
    metrics.reset()

    for param in model.parameters():
        param.requires_grad = True

    return avg_loss, aux


class MACELoss(Metric):
    def __init__(self, loss_fn: torch.nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_data", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("E_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta_es", default=[], dist_reduce_fx="cat")
        self.add_state("delta_es_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Fs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_fs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "stress_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_stress", default=[], dist_reduce_fx="cat")
        self.add_state(
            "virials_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_virials", default=[], dist_reduce_fx="cat")
        self.add_state("delta_virials_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Mus_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus_per_atom", default=[], dist_reduce_fx="cat")

    def update(self, batch, output):  # pylint: disable=arguments-differ
        loss = self.loss_fn(pred=output, ref=batch)
        self.total_loss += loss
        self.num_data += batch.num_graphs

        if output.get("energy") is not None and batch.energy is not None:
            self.E_computed += 1.0
            self.delta_es.append(batch.energy - output["energy"])
            self.delta_es_per_atom.append(
                (batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])
            )
        if output.get("forces") is not None and batch.forces is not None:
            self.Fs_computed += 1.0
            self.fs.append(batch.forces)
            self.delta_fs.append(batch.forces - output["forces"])
        if output.get("stress") is not None and batch.stress is not None:
            self.stress_computed += 1.0
            self.delta_stress.append(batch.stress - output["stress"])
        if output.get("virials") is not None and batch.virials is not None:
            self.virials_computed += 1.0
            self.delta_virials.append(batch.virials - output["virials"])
            self.delta_virials_per_atom.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
        if output.get("dipole") is not None and batch.dipole is not None:
            self.Mus_computed += 1.0
            self.mus.append(batch.dipole)
            self.delta_mus.append(batch.dipole - output["dipole"])
            self.delta_mus_per_atom.append(
                (batch.dipole - output["dipole"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1)
            )

    def convert(self, delta: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return to_numpy(delta)

    def compute(self):
        aux = {}
        aux["loss"] = to_numpy(self.total_loss / self.num_data).item()
        if self.E_computed:
            delta_es = self.convert(self.delta_es)
            delta_es_per_atom = self.convert(self.delta_es_per_atom)
            aux["mae_e"] = compute_mae(delta_es)
            aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
            aux["rmse_e"] = compute_rmse(delta_es)
            aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
            aux["q95_e"] = compute_q95(delta_es)
        if self.Fs_computed:
            fs = self.convert(self.fs)
            delta_fs = self.convert(self.delta_fs)
            aux["mae_f"] = compute_mae(delta_fs)
            aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
            aux["rmse_f"] = compute_rmse(delta_fs)
            aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
            aux["q95_f"] = compute_q95(delta_fs)
        if self.stress_computed:
            delta_stress = self.convert(self.delta_stress)
            aux["mae_stress"] = compute_mae(delta_stress)
            aux["rmse_stress"] = compute_rmse(delta_stress)
            aux["q95_stress"] = compute_q95(delta_stress)
        if self.virials_computed:
            delta_virials = self.convert(self.delta_virials)
            delta_virials_per_atom = self.convert(self.delta_virials_per_atom)
            aux["mae_virials"] = compute_mae(delta_virials)
            aux["rmse_virials"] = compute_rmse(delta_virials)
            aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
            aux["q95_virials"] = compute_q95(delta_virials)
        if self.Mus_computed:
            mus = self.convert(self.mus)
            delta_mus = self.convert(self.delta_mus)
            delta_mus_per_atom = self.convert(self.delta_mus_per_atom)
            aux["mae_mu"] = compute_mae(delta_mus)
            aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
            aux["rel_mae_mu"] = compute_rel_mae(delta_mus, mus)
            aux["rmse_mu"] = compute_rmse(delta_mus)
            aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
            aux["rel_rmse_mu"] = compute_rel_rmse(delta_mus, mus)
            aux["q95_mu"] = compute_q95(delta_mus)

        return aux["loss"], aux
