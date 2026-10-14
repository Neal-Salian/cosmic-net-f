"""
scheduler.py
Purpose: Learning rate schedulers and lambda (λ) annealing for physics-informed loss.
         Implements cosine, linear, and fixed schedules for the virial loss weight.
Inputs: config (dict) - Configuration dictionary from config.yaml
        current_epoch: int - Current training epoch
        total_epochs: int - Total number of training epochs
Outputs: lambda_value: float - Current weight for virial loss term
         lr_scheduler: PyTorch LR scheduler instance
Config keys: training.lambda_schedule, training.lambda_start, training.lambda_end,
             training.lambda_epochs, training.lambda_fixed, training.lr_scheduler,
             training.lr_scheduler_settings
"""

import logging
import math
from typing import Dict, Any, Optional
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    _LRScheduler
)

logger = logging.getLogger(__name__)


class LambdaScheduler:
    """
    Scheduler for the physics loss weight (λ).

    Controls how the virial loss is weighted relative to MSE during training.
    Supports multiple annealing strategies:
    - cosine: Smooth cosine annealing from start to end
    - linear: Linear ramp from start to end
    - fixed: Constant value throughout training

    Gradual annealing allows the model to first fit the data (low λ),
    then progressively enforce physics constraints (high λ).

    training.warmup_epochs holds lambda at zero before the ramp. The ramp
    starts at lambda_start on that epoch and reaches lambda_end after
    lambda_epochs additional epochs. Root warmup_epochs/lambda_end are read
    only when explicitly supplied and their training counterparts are absent;
    they retain the old ramp-to-training.epochs endpoint for compatibility.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the lambda scheduler.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        training_config = config.get('training', {})

        # Schedule type
        self.schedule_type = training_config.get('lambda_schedule', 'cosine')

        # Explicit training settings always win over historical root keys.
        legacy_root = any(key in config and key not in training_config
                          for key in ('warmup_epochs', 'lambda_end'))
        self.warmup_epochs = training_config.get('warmup_epochs', config.get('warmup_epochs', 0))
        self.lambda_start = training_config.get('lambda_start', 0.0)
        self.lambda_end = training_config.get('lambda_end', config.get('lambda_end', 1.0))
        default_ramp = max(1, training_config.get('epochs', 200) - self.warmup_epochs) if legacy_root else 50
        self.lambda_epochs = training_config.get('lambda_epochs', default_ramp)

        # Fixed value (when schedule_type is 'fixed')
        self.lambda_fixed = training_config.get('lambda_fixed', 0.5)

        if self.schedule_type not in {'fixed', 'linear', 'cosine'}:
            raise ValueError(f"unknown training.lambda_schedule: {self.schedule_type!r}")
        for name in ('lambda_start', 'lambda_end', 'lambda_fixed'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"training.{name} must be finite and nonnegative")
        for name, minimum in [('warmup_epochs', 0), ('lambda_epochs', 1)]:
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"training.{name} must be an integer >= {minimum}")
        if legacy_root:
            logger.warning("Using explicitly supplied legacy root lambda settings; prefer training.warmup_epochs/training.lambda_end")

        # Current state
        self.current_epoch = 0
        self.current_lambda = 0.0 if self.warmup_epochs else (self.lambda_fixed if self.schedule_type == 'fixed' else self.lambda_start)

        logger.info(f"LambdaScheduler initialized: type={self.schedule_type}, "
                   f"start={self.lambda_start}, end={self.lambda_end}, "
                   f"epochs={self.lambda_epochs}")

    def step(self, epoch: Optional[int] = None) -> float:
        """
        Update and return the lambda value for the current/given epoch.

        Args:
            epoch: Epoch number (uses internal counter if None)

        Returns:
            Current lambda value
        """
        if epoch is not None:
            if type(epoch) is not int or epoch < 0:
                raise ValueError("lambda scheduler epoch must be a nonnegative integer")
            self.current_epoch = epoch

        if self.current_epoch < self.warmup_epochs:
            self.current_lambda = 0.0

        elif self.schedule_type == 'fixed':
            self.current_lambda = self.lambda_fixed

        elif self.schedule_type == 'linear':
            self.current_lambda = self._linear_schedule()

        elif self.schedule_type == 'cosine':
            self.current_lambda = self._cosine_schedule()

        else:
            logger.warning(f"Unknown schedule type: {self.schedule_type}, using fixed")
            self.current_lambda = self.lambda_fixed

        # Increment epoch if not explicitly provided
        if epoch is None:
            self.current_epoch += 1

        return self.current_lambda

    def _linear_schedule(self) -> float:
        """
        Linear annealing from lambda_start to lambda_end.

        Returns:
            Current lambda value
        """
        ramp_epoch = self.current_epoch - self.warmup_epochs
        if ramp_epoch >= self.lambda_epochs:
            return self.lambda_end

        progress = ramp_epoch / self.lambda_epochs
        return self.lambda_start + progress * (self.lambda_end - self.lambda_start)

    def _cosine_schedule(self) -> float:
        """
        Cosine annealing from lambda_start to lambda_end.

        Smoother than linear, with slower changes at start and end.

        Returns:
            Current lambda value
        """
        ramp_epoch = self.current_epoch - self.warmup_epochs
        if ramp_epoch >= self.lambda_epochs:
            return self.lambda_end

        progress = ramp_epoch / self.lambda_epochs
        cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))
        return self.lambda_start + cosine_factor * (self.lambda_end - self.lambda_start)

    def get_lambda(self) -> float:
        """Get current lambda value without stepping."""
        return self.current_lambda

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            'current_epoch': self.current_epoch,
            'current_lambda': self.current_lambda,
            'schedule_type': self.schedule_type,
            'lambda_start': self.lambda_start,
            'lambda_end': self.lambda_end,
            'lambda_epochs': self.lambda_epochs,
            'lambda_fixed': self.lambda_fixed,
            'warmup_epochs': self.warmup_epochs
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.current_lambda = state_dict.get('current_lambda', self.lambda_start)


class WarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup period.

    Linearly increases LR during warmup, then applies base scheduler.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        base_scheduler: _LRScheduler,
        last_epoch: int = -1
    ):
        """
        Initialize warmup scheduler.

        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            base_scheduler: Scheduler to use after warmup
            last_epoch: Last epoch (for resuming)
        """
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get current learning rates."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            return self.base_scheduler.get_lr()

    def step(self, epoch=None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch >= self.warmup_epochs:
            self.base_scheduler.step(epoch - self.warmup_epochs)

        self._set_lr()

    def _set_lr(self):
        """Set learning rates for all param groups."""
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def build_lr_scheduler(
    optimizer: Optimizer,
    config: Dict[str, Any]
) -> Optional[_LRScheduler]:
    """
    Build learning rate scheduler from config.

    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary

    Returns:
        LR scheduler instance or None
    """
    training_config = config.get('training', {})
    scheduler_type = training_config.get('lr_scheduler', 'cosine')
    settings = training_config.get('lr_scheduler_settings', {})
    total_epochs = training_config.get('epochs', 200)

    if scheduler_type == 'none' or scheduler_type is None:
        return None

    elif scheduler_type == 'cosine':
        T_max = settings.get('T_max', total_epochs)
        eta_min = settings.get('eta_min', 1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    elif scheduler_type == 'step':
        step_size = settings.get('step_size', 50)
        gamma = settings.get('gamma', 0.5)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == 'plateau':
        patience = settings.get('patience', 10)
        factor = settings.get('factor', 0.5)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=patience,
            factor=factor,
            verbose=True
        )

    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, using cosine")
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    logger.info(f"Built LR scheduler: {scheduler_type}")
    return scheduler


def build_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any]
) -> Optimizer:
    """
    Build optimizer from config.

    Args:
        model: PyTorch model
        config: Configuration dictionary

    Returns:
        Optimizer instance
    """
    training_config = config.get('training', {})
    optimizer_type = training_config.get('optimizer', 'adamw')
    lr = training_config.get('learning_rate', 0.001)
    weight_decay = training_config.get('weight_decay', 0.0001)

    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    elif optimizer_type == 'sgd':
        momentum = training_config.get('momentum', 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

    else:
        logger.warning(f"Unknown optimizer: {optimizer_type}, using AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    logger.info(f"Built optimizer: {optimizer_type}, lr={lr}, weight_decay={weight_decay}")
    return optimizer


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.

    Monitors validation loss and stops training if no improvement
    is seen for a specified number of epochs.
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        """
        Initialize early stopping.

        Args:
            patience: Epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
        self.best_epoch = 0

    def step(self, val_loss: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss
            epoch: Current epoch

        Returns:
            True if training should stop
        """
        if self.patience <= 0:
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch}. "
                           f"Best epoch: {self.best_epoch}")

        return self.should_stop

    def state_dict(self) -> Dict[str, Any]:
        """Get state for checkpointing."""
        return {
            'counter': self.counter,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.counter = state_dict.get('counter', 0)
        self.best_loss = state_dict.get('best_loss', float('inf'))
        self.best_epoch = state_dict.get('best_epoch', 0)


class TrainingState:
    """
    Container for all training state (for checkpointing).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        lr_scheduler: Optional[_LRScheduler],
        lambda_scheduler: LambdaScheduler,
        early_stopping: EarlyStopping
    ):
        """
        Initialize training state.

        Args:
            model: The GNN model
            optimizer: PyTorch optimizer
            lr_scheduler: Learning rate scheduler
            lambda_scheduler: Lambda scheduler
            early_stopping: Early stopping handler
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lambda_scheduler = lambda_scheduler
        self.early_stopping = early_stopping
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def state_dict(self) -> Dict[str, Any]:
        """Get complete training state."""
        state = {
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lambda_scheduler_state': self.lambda_scheduler.state_dict(),
            'early_stopping_state': self.early_stopping.state_dict()
        }

        if self.lr_scheduler is not None:
            state['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load complete training state."""
        self.epoch = state_dict.get('epoch', 0)
        self.best_val_loss = state_dict.get('best_val_loss', float('inf'))
        self.train_losses = state_dict.get('train_losses', [])
        self.val_losses = state_dict.get('val_losses', [])

        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.lambda_scheduler.load_state_dict(state_dict['lambda_scheduler_state'])
        self.early_stopping.load_state_dict(state_dict['early_stopping_state'])

        if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in state_dict:
            self.lr_scheduler.load_state_dict(state_dict['lr_scheduler_state_dict'])


def get_current_lr(optimizer: Optimizer) -> float:
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]['lr']
