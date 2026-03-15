"""Physics backend Protocol — abstraction layer for Isaac Gym / Isaac Lab.

Defines the interface that any physics backend must implement. EnvManager
talks to this Protocol, not directly to Isaac Gym or Isaac Lab.

Currently two implementations:
- IsaacGymEnv (IGE_env_manager.py) — Isaac Gym Preview 4, Python 3.8
- IsaacLabEnv (isaac_lab_env_manager.py) — Isaac Lab, Python 3.10+ (stub)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, Tuple

import torch

if TYPE_CHECKING:
    from aerial_gym.env_manager.tensor_state import TensorState


class PhysicsBackend(Protocol):
    """Protocol defining the physics simulation interface.

    Any physics backend (Isaac Gym, Isaac Lab, or future engines) must
    implement these methods. EnvManager depends only on this Protocol.
    """

    # ── Attributes expected on the backend ────────────────────────
    cfg: object  # environment config
    device: str
    num_envs: int
    num_assets_per_env: int
    num_rigid_bodies_robot: Optional[int]
    env_handles: list
    asset_handles: list
    sim_has_dof: bool
    has_IGE_cameras: bool
    env_lower_bound: torch.Tensor
    env_upper_bound: torch.Tensor

    # ── Simulation lifecycle ──────────────────────────────────────

    def create_ground_plane(self) -> None:
        """Add a ground plane to the simulation."""
        ...

    def create_env(self, env_id: int) -> object:
        """Create a single environment instance. Returns an env handle."""
        ...

    def add_asset_to_env(
        self,
        asset_info_dict: dict,
        env_handle: object,
        env_id: int,
        global_asset_counter: int,
        segmentation_counter: int,
    ) -> Tuple[object, int]:
        """Add an asset (robot, obstacle, wall) to an environment.

        Returns (asset_handle, segmentation_counter_increment).
        """
        ...

    def prepare_for_simulation(
        self,
        env_manager: object,
        global_tensor_dict: TensorState,
    ) -> bool:
        """Finalize simulation setup: acquire tensors, create viewer.

        Called after all environments and assets are created.
        Returns True on success.
        """
        ...

    # ── Physics step cycle ────────────────────────────────────────

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply forces/torques before physics simulation step."""
        ...

    def physics_step(self) -> None:
        """Run one physics simulation step."""
        ...

    def post_physics_step(self) -> None:
        """Synchronize state tensors after physics step."""
        ...

    # ── State management ──────────────────────────────────────────

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset specified environments (randomize bounds, etc.)."""
        ...

    def write_to_sim(self) -> None:
        """Write state tensors back to the simulation."""
        ...

    def refresh_tensors(self) -> None:
        """Refresh all state tensors from the simulation."""
        ...

    # ── Rendering ─────────────────────────────────────────────────

    def step_graphics(self) -> None:
        """Update graphics pipeline (required before camera capture)."""
        ...

    def render_viewer(self) -> None:
        """Render the viewer window (no-op in headless mode)."""
        ...

    # ── Viewer ────────────────────────────────────────────────────

    def create_viewer(self, env_manager: object) -> None:
        """Create visualization window (no-op in headless mode)."""
        ...
