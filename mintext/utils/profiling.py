"""JAX profiler wrapper."""

from __future__ import annotations

import logging
from pathlib import Path

import jax

from mintext.config import MinTextConfig

logger = logging.getLogger(__name__)


class Profiler:
    """JAX profiler that activates for a window of steps.

    Usage:
        prof = Profiler(config)
        for step in range(steps):
            prof.maybe_activate(step)
            # ... train step ...
            prof.maybe_deactivate(step)
    """

    def __init__(self, config: MinTextConfig):
        self.enabled = config.enable_profiler
        self.skip_steps = config.skip_first_n_profiler_steps
        self.profile_steps = config.profiler_steps
        self.start_step = self.skip_steps
        self.end_step = self.skip_steps + self.profile_steps
        self.active = False

        if self.enabled:
            self.output_dir = str(
                Path(config.base_output_directory) / config.run_name / "profiles"
            )
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            logger.info(
                "Profiler enabled: steps %d-%d, output %s",
                self.start_step, self.end_step - 1, self.output_dir,
            )

    def maybe_activate(self, step: int) -> None:
        """Start profiling if this is the start step."""
        if not self.enabled or self.active:
            return
        if step == self.start_step:
            jax.profiler.start_trace(self.output_dir)
            self.active = True
            logger.info("Profiler started at step %d", step)

    def maybe_deactivate(self, step: int) -> None:
        """Stop profiling if this is the end step."""
        if not self.active:
            return
        if step >= self.end_step - 1:
            jax.profiler.stop_trace()
            self.active = False
            logger.info("Profiler stopped at step %d, trace saved to %s", step, self.output_dir)

    def stop(self) -> None:
        """Force stop if still active (e.g., training ended early)."""
        if self.active:
            jax.profiler.stop_trace()
            self.active = False
