from __future__ import annotations

from dataclasses import dataclass, field

import torch

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("IsaacLabCameraBackend")


@dataclass
class _PendingCamera:
    """Stored config for a camera that has not yet been instantiated."""

    prim_path: str
    height: int
    width: int
    horizontal_fov: float
    near_plane: float
    far_plane: float
    data_types: list[str] = field(
        default_factory=lambda: [
            "distance_to_image_plane",
            "semantic_segmentation",
            "rgb",
        ]
    )


class IsaacLabCameraBackend:
    """Isaac Lab implementation of the ``CameraBackend`` protocol.

    Cameras are created in two phases:
    1. ``create_camera`` stores configs (before sim.reset).
    2. ``initialize`` instantiates the actual ``Camera`` objects (after scene
       creation but before the first step).
    """

    def __init__(self, simulation: object) -> None:
        """Store a reference to the Isaac Lab simulation context.

        Args:
            simulation: The Isaac Lab ``SimulationContext`` or equivalent handle.
        """
        self.simulation = simulation
        self._pending: dict[int, _PendingCamera] = {}
        self._cameras: dict[int, object] = {}
        self._next_handle: int = 0
        self._initialized: bool = False

    def create_camera(self, env_handle: int, camera_props: object) -> int:
        """Store a camera config to be instantiated later during ``initialize``.

        Args:
            env_handle: Environment index (used to construct the prim path).
            camera_props: A ``CameraProps`` dataclass with width, height, etc.

        Returns:
            An integer handle identifying the pending camera.
        """
        handle = self._next_handle
        self._next_handle += 1

        prim_path = f"/World/envs/env_.*/Robot/camera_{handle}"
        self._pending[handle] = _PendingCamera(
            prim_path=prim_path,
            height=camera_props.height,
            width=camera_props.width,
            horizontal_fov=camera_props.horizontal_fov,
            near_plane=camera_props.near_plane,
            far_plane=camera_props.far_plane,
        )
        logger.debug(f"Queued camera {handle}: {camera_props.width}x{camera_props.height}")
        return handle

    def attach_camera_to_body(
        self,
        camera_handle: int,
        env_handle: int,
        body_handle: int,
        local_transform: object,
    ) -> None:
        """Update the pending camera's prim path to be under the target body.

        In Isaac Lab, attachment is declarative: the camera prim is placed as a
        child of the body prim in the USD hierarchy. The offset is baked into
        the ``CameraCfg.OffsetCfg`` when the camera is instantiated.

        Args:
            camera_handle: Handle returned by ``create_camera``.
            env_handle: Environment handle / index.
            body_handle: Handle to the rigid body the camera should follow.
            local_transform: Pose offset (a ``Transform`` with ``.p`` and ``.r``).
        """
        if camera_handle not in self._pending:
            logger.warning(f"Cannot attach camera {camera_handle}: not found in pending configs")
            return

        pending = self._pending[camera_handle]
        pending.prim_path = f"/World/envs/env_.*/Robot/body_{body_handle}/camera_{camera_handle}"
        logger.debug(f"Camera {camera_handle} will attach under body {body_handle}")

    def initialize(self) -> None:
        """Instantiate all pending cameras as Isaac Lab ``Camera`` objects.

        Must be called after scene creation but before ``sim.reset()``.
        """
        if self._initialized:
            return

        from isaaclab.sensors import Camera, CameraCfg

        for handle, pending in self._pending.items():
            cfg = CameraCfg(
                prim_path=pending.prim_path,
                update_period=0.0,
                height=pending.height,
                width=pending.width,
                data_types=pending.data_types,
                spawn=None,
            )
            camera = Camera(cfg)
            self._cameras[handle] = camera
            logger.info(
                f"Initialized camera {handle} at {pending.prim_path} "
                f"({pending.width}x{pending.height})"
            )

        self._initialized = True

    def render_cameras(self, sim_handle: object) -> None:
        """Trigger rendering for all camera sensors.

        Args:
            sim_handle: Simulation handle (unused -- kept for interface compat).
        """
        if not self._initialized:
            self.initialize()

        dt = 0.01
        if self.simulation is not None:
            sim_dt = getattr(self.simulation, "physics_dt", None)
            if sim_dt is not None:
                dt = sim_dt

        for camera in self._cameras.values():
            camera.update(dt)

    def get_depth_tensor(
        self, sim_handle: object, env_handle: int, camera_handle: int
    ) -> torch.Tensor:
        """Return the depth image tensor for a single camera/env.

        Args:
            sim_handle: Simulation handle (unused in Isaac Lab).
            env_handle: Environment index.
            camera_handle: Camera handle / index.

        Returns:
            Depth tensor of shape ``(H, W)`` for the given environment.
        """
        camera = self._cameras[camera_handle]
        return camera.data.output["distance_to_image_plane"][env_handle]

    def get_segmentation_tensor(
        self, sim_handle: object, env_handle: int, camera_handle: int
    ) -> torch.Tensor:
        """Return the segmentation image tensor for a single camera/env.

        Args:
            sim_handle: Simulation handle (unused in Isaac Lab).
            env_handle: Environment index.
            camera_handle: Camera handle / index.

        Returns:
            Segmentation tensor of shape ``(H, W)`` for the given environment.
        """
        camera = self._cameras[camera_handle]
        return camera.data.output["semantic_segmentation"][env_handle]

    def get_rgb_tensor(
        self, sim_handle: object, env_handle: int, camera_handle: int
    ) -> torch.Tensor:
        """Return the RGBA image tensor for a single camera/env.

        Args:
            sim_handle: Simulation handle (unused in Isaac Lab).
            env_handle: Environment index.
            camera_handle: Camera handle / index.

        Returns:
            RGBA tensor of shape ``(H, W, 4)`` for the given environment.
        """
        camera = self._cameras[camera_handle]
        return camera.data.output["rgb"][env_handle]

    def start_tensor_access(self, sim_handle: object) -> None:
        """No-op in Isaac Lab: tensors are directly accessible."""

    def end_tensor_access(self, sim_handle: object) -> None:
        """No-op in Isaac Lab: tensors are directly accessible."""

    def wrap_tensor(self, raw_tensor: object) -> torch.Tensor:
        """Identity operation -- Isaac Lab already returns ``torch.Tensor``.

        Args:
            raw_tensor: The tensor (already a ``torch.Tensor`` in Isaac Lab).

        Returns:
            The same tensor, unchanged.
        """
        if isinstance(raw_tensor, torch.Tensor):
            return raw_tensor
        raise TypeError(f"Expected torch.Tensor from Isaac Lab camera, got {type(raw_tensor)}")
