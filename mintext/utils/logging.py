"""TensorBoard logging and training metrics."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from mintext.config import MinTextConfig
from mintext.utils.pytree import calculate_tflops_per_device, calculate_tokens_per_device, count_params

logger = logging.getLogger(__name__)


class MetricLogger:
    """Training metric logger with TensorBoard and console output.

    Logs scalar metrics (loss, lr, grad_norm, throughput, TFLOP/s) to
    TensorBoard and console. Only process 0 writes to disk.
    """

    def __init__(self, config: MinTextConfig, params: Any = None):
        self.tflops_per_device = calculate_tflops_per_device(config)
        self.tokens_per_device = calculate_tokens_per_device(config)
        self.writer = None

        # Setup TensorBoard
        if config.enable_tensorboard:
            import jax
            if jax.process_index() == 0:
                try:
                    from tensorboardX import SummaryWriter
                    tb_dir = config.tensorboard_dir or str(
                        Path(config.base_output_directory) / config.run_name / "tensorboard"
                    )
                    Path(tb_dir).mkdir(parents=True, exist_ok=True)
                    self.writer = SummaryWriter(tb_dir)
                    logger.info("TensorBoard logging to %s", tb_dir)
                except ImportError:
                    logger.warning("tensorboardX not installed, TensorBoard disabled")

        # Log model info at startup
        if params is not None:
            n_params = count_params(params)
            logger.info("Model parameters: %s (%.2fM)", f"{n_params:,}", n_params / 1e6)
            logger.info("Estimated TFLOP/device/step: %.2f", self.tflops_per_device)
            logger.info("Tokens/device/step: %d", self.tokens_per_device)
            if self.writer is not None:
                self.writer.add_text("config/model_params", f"{n_params:,}")
                self.writer.add_text("config/model_name", config.model_name)

    def log_step(
        self,
        step: int,
        metrics: dict[str, Any],
        step_time: float,
    ) -> None:
        """Log training step metrics to TensorBoard and console.

        Args:
            step: Current training step.
            metrics: Dict with 'loss', 'learning_rate', 'grad_norm', etc.
            step_time: Wall-clock time for this step in seconds.
        """
        loss = float(metrics["loss"])
        lr = float(metrics["learning_rate"])
        grad_norm = float(metrics["grad_norm"])

        # Compute throughput
        tflops_per_sec = self.tflops_per_device / step_time if step_time > 0 else 0.0
        tokens_per_sec = self.tokens_per_device / step_time if step_time > 0 else 0.0

        # Console log
        logger.info(
            "step=%d loss=%.4f lr=%.2e grad_norm=%.4f "
            "step_time=%.3fs TFLOP/s/device=%.1f tokens/s/device=%.0f",
            step, loss, lr, grad_norm, step_time, tflops_per_sec, tokens_per_sec,
        )

        # TensorBoard
        if self.writer is not None:
            self.writer.add_scalar("learning/loss", loss, step)
            self.writer.add_scalar("learning/learning_rate", lr, step)
            self.writer.add_scalar("learning/grad_norm", grad_norm, step)
            if "param_norm" in metrics:
                self.writer.add_scalar("learning/param_norm", float(metrics["param_norm"]), step)
            self.writer.add_scalar("perf/step_time_seconds", step_time, step)
            self.writer.add_scalar("perf/tflops_per_sec_per_device", tflops_per_sec, step)
            self.writer.add_scalar("perf/tokens_per_sec_per_device", tokens_per_sec, step)

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None
