"""
main.py
Purpose: Main entry point for Cosmic-Net pipeline. Orchestrates data loading,
         graph construction, model training, evaluation, explainability, and
         symbolic regression in a unified workflow.
Inputs: config/config.yaml - Configuration file
        Command line arguments for mode selection
Outputs: Trained model checkpoints, W&B logs, explanations, discovered equations
Config keys: All configuration is loaded from config/config.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch
import numpy as np
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create log directory
    log_file = log_config.get('log_file', 'outputs/logs/cosmic_net.log')
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_output_dirs() -> None:
    """Create required output directories."""
    dirs = [
        'outputs/checkpoints',
        'outputs/explanations',
        'outputs/equations',
        'outputs/logs'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def train(config: Dict[str, Any]) -> None:
    """
    Run the full training pipeline.

    Steps:
    1. Load data from configured source
    2. Build graphs
    3. Train GNN with physics-informed loss
    4. Evaluate on test set
    5. Log results to W&B
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("COSMIC-NET TRAINING PIPELINE")
    logger.info("=" * 60)

    # Import modules
    from data.loaders.base_loader import get_loader, get_train_test_loaders
    from graph.graph_builder import GraphBuilder, build_dataloaders
    from model.model import build_model
    from training.train import train_model

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Check for cross-simulation experiment
    train_source = config.get('training', {}).get('train_source', 'synthetic')
    test_source = config.get('training', {}).get('test_source', 'synthetic')

    if train_source != test_source:
        raise ValueError("Legacy cross-simulation training cannot certify compatible targets, "
                         "units or independent splits. Use the audited astronomy baseline "
                         "and evaluation workflow with explicit catalog contracts.")
    else:
        # Standard single-source experiment
        loader = get_loader(config)
        train_halos, val_halos, test_halos = loader.split_data()

    logger.info(f"Data split: train={len(train_halos)}, val={len(val_halos)}, test={len(test_halos)}")

    # Build data loaders
    train_loader, val_loader, test_loader = build_dataloaders(
        config, train_halos, val_halos, test_halos
    )

    # Train model
    model, results = train_model(config, train_loader, val_loader, test_loader)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best validation MSE: {results['best_val_loss']:.4f}")
    logger.info(f"Best epoch: {results['best_epoch']}")
    if results.get('final_test_metrics'):
        logger.info(f"Test MSE: {results['final_test_metrics'].get('test/mse', 'N/A'):.4f}")
        logger.info(f"Test R²: {results['final_test_metrics'].get('test/r2', 'N/A'):.4f}")
    logger.info("=" * 60)


def evaluate(config: Dict[str, Any], checkpoint_path: Optional[str] = None) -> None:
    """
    Evaluate a trained model on test data.

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint (uses best_model.pt if None)
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("COSMIC-NET EVALUATION")
    logger.info("=" * 60)

    from data.loaders.base_loader import get_loader
    from graph.graph_builder import GraphBuilder, build_dataloaders
    from model.model import load_model
    from model.physics_loss import MetricsComputer
    from training.train import Trainer

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    loader = get_loader(config)
    _, _, test_halos = loader.split_data()

    # Build test loader
    graph_builder = GraphBuilder(config)
    test_graphs = graph_builder.build_graphs(test_halos)

    from torch_geometric.loader import DataLoader
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    # Load model
    if checkpoint_path is None:
        checkpoint_path = config.get('training', {}).get('checkpoint_dir', 'outputs/checkpoints')
        checkpoint_path = Path(checkpoint_path) / 'best_model.pt'

    model = load_model(str(checkpoint_path), config, device)

    # Evaluate with uncertainty
    model.eval()
    all_preds = []
    all_targets = []
    all_stds = []

    for batch_data in test_loader:
        batch_data = batch_data.to(device)

        # Add required attributes if missing
        if not hasattr(batch_data, 'stellar_mass'):
            batch_data.stellar_mass = torch.ones(batch_data.x.size(0), device=device) * 1e10
        if not hasattr(batch_data, 'vel_disp'):
            batch_data.vel_disp = torch.ones(batch_data.x.size(0), device=device) * 100
        if not hasattr(batch_data, 'half_mass_r'):
            batch_data.half_mass_r = torch.ones(batch_data.x.size(0), device=device) * 0.01

        uncertainty = model.predict_with_uncertainty(batch_data)
        all_preds.append(uncertainty['mean'].cpu())
        all_stds.append(uncertainty['std'].cpu())
        all_targets.append(batch_data.y.cpu())

    all_preds = torch.cat(all_preds)
    all_stds = torch.cat(all_stds)
    all_targets = torch.cat(all_targets)

    # Compute metrics
    metrics = MetricsComputer.compute_all(all_preds, all_targets)

    logger.info("Evaluation Results:")
    logger.info(f"  MSE: {metrics['mse']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    logger.info(f"  Scatter: {metrics['scatter']:.4f}")
    logger.info(f"  Mean uncertainty: {all_stds.mean():.4f}")

    # Calibration
    conf_low = all_preds - 1.96 * all_stds
    conf_high = all_preds + 1.96 * all_stds
    coverage = ((all_targets >= conf_low) & (all_targets <= conf_high)).float().mean()
    logger.info(f"  95% CI coverage: {coverage:.2%}")

    logger.info("=" * 60)


def explain(config: Dict[str, Any], checkpoint_path: Optional[str] = None) -> None:
    """
    Generate explanations for test set predictions.

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("COSMIC-NET EXPLAINABILITY")
    logger.info("=" * 60)

    from data.loaders.base_loader import get_loader
    from graph.graph_builder import GraphBuilder
    from model.model import load_model
    from explain.explainer import create_explainer

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data (use a subset for explanation)
    loader = get_loader(config)
    _, _, test_halos = loader.split_data()
    test_halos = test_halos[:10]  # Explain first 10 for demo

    # Build graphs
    graph_builder = GraphBuilder(config)
    test_graphs = graph_builder.build_graphs(test_halos)

    # Load model
    if checkpoint_path is None:
        checkpoint_path = config.get('training', {}).get('checkpoint_dir', 'outputs/checkpoints')
        checkpoint_path = Path(checkpoint_path) / 'best_model.pt'

    model = load_model(str(checkpoint_path), config, device)

    # Create explainer
    explainer = create_explainer(model, config, device)

    # Generate explanations
    results = []
    for graph in test_graphs:
        graph = graph.to(device)
        try:
            result = explainer.explain(graph)
            results.append(result)
            logger.info(f"Explained {result.cluster_id}: prediction={result.prediction:.2f}")
        except Exception as e:
            logger.warning(f"Failed to explain graph: {e}")

    # Save all explanations
    explainer.save_all_explanations(results)

    logger.info(f"Generated {len(results)} explanations")
    logger.info("=" * 60)


def symbolic(config: Dict[str, Any], checkpoint_path: Optional[str] = None) -> None:
    """
    Run symbolic regression to discover equations.

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
    """
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("COSMIC-NET SYMBOLIC REGRESSION")
    logger.info("=" * 60)

    from data.loaders.base_loader import get_loader
    from graph.graph_builder import GraphBuilder
    from model.model import load_model
    from symbolic.symbolic_regression import run_symbolic_regression
    from torch_geometric.loader import DataLoader

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    loader = get_loader(config)
    train_halos, _, _ = loader.split_data()

    # Build graphs
    graph_builder = GraphBuilder(config)
    train_graphs = graph_builder.build_graphs(train_halos)
    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=False)

    # Load model
    if checkpoint_path is None:
        checkpoint_path = config.get('training', {}).get('checkpoint_dir', 'outputs/checkpoints')
        checkpoint_path = Path(checkpoint_path) / 'best_model.pt'

    model = load_model(str(checkpoint_path), config, device)

    # Run symbolic regression
    regressor = run_symbolic_regression(model, train_loader, config, device)

    # Report results
    best_eq = regressor.get_best_equation()
    if best_eq:
        logger.info(f"Best equation: {best_eq.equation_str}")
        logger.info(f"  RMSE: {best_eq.rmse:.4f}")
        logger.info(f"  Complexity: {best_eq.complexity}")
        logger.info(f"  Dimensionally valid: {best_eq.is_dimensionally_valid}")
        logger.info(f"  Notes: {best_eq.dimension_check_notes}")

    logger.info(f"Total equations discovered: {len(regressor.equations)}")
    logger.info(f"Dimensionally valid: {len(regressor.pareto_front)}")
    logger.info("=" * 60)


def serve(config: Dict[str, Any]) -> None:
    """
    Start the FastAPI server.

    Args:
        config: Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Cosmic-Net API server...")

    from deploy.api import run_server

    api_config = config.get('api', {})
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)

    run_server(host=host, port=port)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Cosmic-Net: Physics-Informed GNN for Dark Matter Halo Mass Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train              # Train the model
  python main.py evaluate           # Evaluate trained model
  python main.py explain            # Generate explanations
  python main.py symbolic           # Run symbolic regression
  python main.py serve              # Start API server
  python main.py train --config custom.yaml   # Use custom config
        """
    )

    parser.add_argument(
        'mode',
        choices=['train', 'evaluate', 'explain', 'symbolic', 'serve', 'full'],
        help='Pipeline mode to run'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Path to model checkpoint (for evaluate/explain/symbolic)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Override random seed from config'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override seed if provided
    if args.seed is not None:
        config['seed'] = args.seed

    # Override log level if debug
    if args.debug:
        config['logging'] = config.get('logging', {})
        config['logging']['level'] = 'DEBUG'

    # Setup
    setup_logging(config)
    set_seed(config.get('seed', 42))
    create_output_dirs()

    logger = logging.getLogger(__name__)
    logger.info(f"Cosmic-Net v1.0.0 | Mode: {args.mode} | Seed: {config.get('seed', 42)}")

    # Run selected mode
    if args.mode == 'train':
        train(config)
    elif args.mode == 'evaluate':
        evaluate(config, args.checkpoint)
    elif args.mode == 'explain':
        explain(config, args.checkpoint)
    elif args.mode == 'symbolic':
        symbolic(config, args.checkpoint)
    elif args.mode == 'serve':
        serve(config)
    elif args.mode == 'full':
        # Run full pipeline: train -> evaluate -> explain -> symbolic
        logger.info("Running full pipeline...")
        train(config)
        evaluate(config)
        explain(config)
        symbolic(config)
        logger.info("Full pipeline complete!")


if __name__ == "__main__":
    main()
