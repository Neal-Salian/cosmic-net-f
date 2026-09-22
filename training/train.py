"""
train.py
Purpose: Main training loop for Cosmic-Net GNN with W&B logging, checkpointing,
         and support for cross-simulation generalization experiments.
"""

import os
import logging
import time
import math
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

from dotenv import load_dotenv

from model.model import CosmicNetGNN, build_model
from model.physics_loss import PhysicsInformedLoss, MetricsComputer
from training.scheduler import (
    build_optimizer,
    build_lr_scheduler,
    EarlyStopping,
    TrainingState,
    get_current_lr
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CosineLambdaScheduler:
    """
    Custom strictly-enforced Lambda Scheduler for Physics Loss.
    Guarantees lambda stays strictly at 0.0 during warmup, then
    applies a smooth cosine annealing curve up to lambda_end.
    """
    def __init__(self, warmup_epochs: int, lambda_end: float, total_epochs: int):
        self.warmup_epochs = warmup_epochs
        self.lambda_end = lambda_end
        self.total_epochs = total_epochs
        self.current_lambda = 0.0

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            self.current_lambda = 0.0
        else:
            # Calculate progress from 0.0 to 1.0 AFTER warmup
            progress = (epoch - self.warmup_epochs) / max(1, (self.total_epochs - self.warmup_epochs))
            progress = min(1.0, max(0.0, progress)) # Clamp just in case
            
            # Cosine curve from 0 to lambda_end
            self.current_lambda = self.lambda_end * 0.5 * (1 - math.cos(math.pi * progress))
            
        return self.current_lambda

    def get_lambda(self) -> float:
        return self.current_lambda

    def state_dict(self) -> Dict[str, Any]:
        return {'current_lambda': self.current_lambda}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.current_lambda = state.get('current_lambda', 0.0)


class Trainer:
    """
    Main trainer class for Cosmic-Net GNN.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: CosmicNetGNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.training_config = config.get('training', {})
        self.wandb_config = config.get('wandb', {})

        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Model
        self.model = model.to(self.device)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Loss function
        self.loss_fn = PhysicsInformedLoss(config)

        # Optimizer and schedulers
        self.optimizer = build_optimizer(self.model, config)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, config)
        
        # --- NEW: Inject the custom Cosine Scheduler here ---
        # Pull parameters from the root config, with fail-safes
        warmup = config.get('warmup_epochs', 75)
        l_end = config.get('lambda_end', 0.005)
        epochs = self.training_config.get('epochs', 500)
        
        self.lambda_scheduler = CosineLambdaScheduler(
            warmup_epochs=warmup, 
            lambda_end=l_end, 
            total_epochs=epochs
        )
        logger.info(f"Initialized CosineLambdaScheduler: Warmup={warmup}, MaxLambda={l_end}")
        # ----------------------------------------------------

        # Early stopping
        patience = self.training_config.get('early_stopping_patience', 20)
        self.early_stopping = EarlyStopping(patience=patience)

        # Training state
        self.state = TrainingState(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            lambda_scheduler=self.lambda_scheduler,
            early_stopping=self.early_stopping
        )

        # Gradient clipping
        self.grad_clip = self.training_config.get('grad_clip', 1.0)

        # Checkpointing
        self.checkpoint_dir = Path(self.training_config.get('checkpoint_dir', 'outputs/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = self.training_config.get('save_every', 10)
        self.save_best = self.training_config.get('save_best', True)

        # Cross-simulation experiment detection
        train_source = self.training_config.get('train_source', 'synthetic')
        test_source = self.training_config.get('test_source', 'synthetic')
        self.is_cross_sim = train_source != test_source
        if self.is_cross_sim:
            logger.info(f"Cross-simulation experiment: train on {train_source}, test on {test_source}")

        # W&B initialization
        self.use_wandb = self.wandb_config.get('enabled', True) and HAS_WANDB
        self._init_wandb()

        logger.info(f"Trainer initialized on device: {self.device}")

    def _init_wandb(self) -> None:
        if not self.use_wandb:
            logger.info("W&B logging disabled")
            return

        try:
            api_key = os.environ.get('WANDB_API_KEY')
            if api_key:
                wandb.login(key=api_key)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prefix = self.wandb_config.get('run_name_prefix', 'cosmicnet')
            run_name = f"{prefix}_{timestamp}"

            tags = list(self.wandb_config.get('tags', ['gnn', 'physics-informed']))
            if self.is_cross_sim:
                tags.append('cross_sim')

            wandb.init(
                project=self.wandb_config.get('project', 'cosmic-net'),
                entity=self.wandb_config.get('entity') or os.environ.get('WANDB_ENTITY'),
                name=run_name,
                config=self.config,
                tags=tags,
                reinit=True
            )

            if self.wandb_config.get('watch_model', True):
                wandb.watch(
                    self.model,
                    log='all',
                    log_freq=self.wandb_config.get('watch_freq', 100)
                )

            logger.info(f"W&B initialized: {wandb.run.name}")

        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.use_wandb = False

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        total_mse = 0.0
        total_virial = 0.0
        num_batches = 0

        lambda_value = self.lambda_scheduler.step(epoch)

        for batch_data in self.train_loader:
            batch_data = batch_data.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions, _ = self.model(batch_data)

            # Compute loss
            targets = batch_data.y
            loss, loss_dict = self.loss_fn(predictions, targets, batch_data, lambda_value)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # Optimizer step
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss_dict['total_loss'].item()
            total_mse += loss_dict['mse_loss'].item()
            total_virial += loss_dict['virial_loss'].item()
            num_batches += 1

        # Average metrics
        metrics = {
            'train/loss': total_loss / max(num_batches, 1),
            'train/mse': total_mse / max(num_batches, 1),
            'train/virial_loss': total_virial / max(num_batches, 1),
            'train/lambda': lambda_value,
            'train/lr': get_current_lr(self.optimizer)
        }

        return metrics

    @torch.no_grad()
    def validate(self, loader: DataLoader, prefix: str = 'val') -> Dict[str, float]:
        self.model.eval()

        all_predictions = []
        all_targets = []
        total_loss = 0.0
        total_mse = 0.0
        total_virial = 0.0
        num_batches = 0

        lambda_value = self.lambda_scheduler.get_lambda()

        for batch_data in loader:
            batch_data = batch_data.to(self.device)

            # Forward pass
            predictions, _ = self.model(batch_data)
            targets = batch_data.y

            # Compute loss
            loss, loss_dict = self.loss_fn(predictions, targets, batch_data, lambda_value)

            # Accumulate
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            total_loss += loss_dict['total_loss'].item()
            total_mse += loss_dict['mse_loss'].item()
            total_virial += loss_dict['virial_loss'].item()
            num_batches += 1

        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        detailed_metrics = MetricsComputer.compute_all(all_predictions, all_targets)

        metrics = {
            f'{prefix}/loss': total_loss / max(num_batches, 1),
            f'{prefix}/mse': total_mse / max(num_batches, 1),
            f'{prefix}/virial_loss': total_virial / max(num_batches, 1),
            f'{prefix}/rmse': detailed_metrics['rmse'],
            f'{prefix}/mae': detailed_metrics['mae'],
            f'{prefix}/r2': detailed_metrics['r2'],
            f'{prefix}/scatter': detailed_metrics['scatter']
        }

        return metrics

    def train(self) -> Dict[str, Any]:
        epochs = self.training_config.get('epochs', 200)

        logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()

        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(self.val_loader, prefix='val')

            # Combine metrics
            metrics = {**train_metrics, **val_metrics, 'epoch': epoch}

            # LR scheduler step
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_metrics['val/mse'])
                else:
                    self.lr_scheduler.step()

            # Track best model
            current_val_loss = val_metrics['val/mse']
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch = epoch
                if self.save_best:
                    self._save_checkpoint('best_model.pt', epoch, metrics)

            # Early stopping
            if self.early_stopping.step(current_val_loss, epoch):
                logger.info(f"Early stopping at epoch {epoch}")
                break

            # Periodic checkpoint
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch, metrics)

            # Log to W&B
            if self.use_wandb:
                metrics.update({f'weights/{k}': v for k, v in self.model.get_weight_norms().items()})
                wandb.log(metrics, step=epoch)

            # Log to console
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train MSE: {train_metrics['train/mse']:.4f} | "
                f"Val MSE: {val_metrics['val/mse']:.4f} | "
                f"Val R²: {val_metrics['val/r2']:.4f} | "
                f"λ: {train_metrics['train/lambda']:.3f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Store losses
            self.state.train_losses.append(train_metrics['train/mse'])
            self.state.val_losses.append(val_metrics['val/mse'])
            self.state.epoch = epoch

        # Training complete
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.1f} minutes")
        logger.info(f"Best validation MSE: {best_val_loss:.4f} at epoch {best_epoch}")

        # Save final checkpoint with last-epoch weights and validation metrics
        # BEFORE reloading best weights for test evaluation.
        final_metrics = {**val_metrics}
        self._save_checkpoint('final_model.pt', epoch, final_metrics)

        # Final test evaluation using best weights
        test_metrics = {}
        if self.test_loader is not None:
            best_checkpoint = self.checkpoint_dir / 'best_model.pt'
            if best_checkpoint.exists():
                self._load_checkpoint(best_checkpoint)

            test_metrics = self.validate(self.test_loader, prefix='test')
            logger.info(f"Test MSE: {test_metrics['test/mse']:.4f} | "
                       f"Test R²: {test_metrics['test/r2']:.4f}")

            if self.use_wandb:
                wandb.log(test_metrics)

        if self.use_wandb:
            wandb.finish()

        return {
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'final_test_metrics': test_metrics,
            'total_time': total_time
        }

    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float]) -> None:
        from data.provenance import model_state_hash, validate_checkpoint_file
        state_hash = model_state_hash(self.model.state_dict())
        provenance = deepcopy(self.config.get('provenance', {
            'schema_version': 1,
            'research_label': self.config.get('data', {}).get(
                'research_label', 'legacy_uncertified'),
        }))
        provenance['model_state_sha256'] = state_hash
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lambda_scheduler_state': self.lambda_scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'provenance': provenance,
        }

        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        checkpoint_info = validate_checkpoint_file(path)
        manifest = {
            **provenance,
            'checkpoint_path': str(path),
            'checkpoint_sha256': checkpoint_info['sha256'],
            'checkpoint_size_bytes': checkpoint_info['size_bytes'],
        }
        path.with_name(path.name + '.manifest.json').write_text(
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False)
        )
        logger.debug(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, path: Path) -> None:
        from data.provenance import validate_checkpoint_file, validate_run_provenance
        validate_checkpoint_file(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get('provenance'):
            validate_run_provenance(
                checkpoint['provenance'], model_state=checkpoint['model_state_dict']
            )
        active_provenance = self.config.get('provenance')
        ckpt_provenance = checkpoint.get('provenance') if isinstance(checkpoint, dict) else None
        if active_provenance and not ckpt_provenance:
            raise ValueError(
                "Active run requires provenance but checkpoint has no provenance; "
                "refusing to load unprovenanced weights into a provenanced run"
            )
        if active_provenance and ckpt_provenance:
            for key in ('catalog_contract_sha256', 'split_manifest_sha256',
                        'config_sha256', 'research_label'):
                if active_provenance.get(key) != ckpt_provenance.get(key):
                    raise ValueError(
                        f"checkpoint provenance mismatch for {key}: "
                        f"active={active_provenance.get(key)!r} "
                        f"vs checkpoint={ckpt_provenance.get(key)!r}"
                    )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint: {path}")

    @torch.no_grad()
    def evaluate_with_uncertainty(self, loader: DataLoader, n_samples: int = 50) -> Dict[str, Any]:
        all_means = []
        all_stds = []
        all_targets = []
        all_cluster_ids = []

        for batch_data in loader:
            batch_data = batch_data.to(self.device)
            uncertainty = self.model.predict_with_uncertainty(batch_data, n_samples)

            all_means.append(uncertainty['mean'].cpu())
            all_stds.append(uncertainty['std'].cpu())
            all_targets.append(batch_data.y.cpu())

            if hasattr(batch_data, 'cluster_id'):
                all_cluster_ids.extend(batch_data.cluster_id)

        all_means = torch.cat(all_means)
        all_stds = torch.cat(all_stds)
        all_targets = torch.cat(all_targets)

        metrics = MetricsComputer.compute_all(all_means, all_targets)

        conf_low = all_means - 1.96 * all_stds
        conf_high = all_means + 1.96 * all_stds
        coverage = ((all_targets >= conf_low) & (all_targets <= conf_high)).float().mean()

        return {
            'predictions': all_means.numpy(),
            'uncertainties': all_stds.numpy(),
            'targets': all_targets.numpy(),
            'cluster_ids': all_cluster_ids,
            'metrics': metrics,
            'coverage_95': coverage.item()
        }


def train_model(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None
) -> Tuple[CosmicNetGNN, Dict[str, Any]]:
    model = build_model(config)
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    results = trainer.train()
    return model, results


def run_ablation_study(
    config: Dict[str, Any],
    ablation_configs: List[Dict[str, Any]],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None
) -> List[Dict[str, Any]]:
    results = []

    for i, ablation in enumerate(ablation_configs):
        run_config = {**config}
        for key, value in ablation.items():
            keys = key.split('.')
            d = run_config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

        logger.info(f"Running ablation {i+1}/{len(ablation_configs)}: {ablation}")

        model, run_results = train_model(
            run_config, train_loader, val_loader, test_loader
        )

        results.append({
            'ablation': ablation,
            'results': run_results
        })

    return results
