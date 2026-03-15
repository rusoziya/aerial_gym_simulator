from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class CameraBackend(Protocol):
    """Protocol defining the camera sensor backend interface.

    Any physics engine backend (Isaac Gym, Isaac Lab, etc.) must implement
    these methods to be used as a camera sensor provider. The interface
    covers the full lifecycle: creation, attachment, rendering, and
    GPU tensor access for depth, segmentation, and RGB images.
    """

    def create_camera(self, env_handle: int, camera_props: object) -> int:
        """Create a camera sensor in the given environment.

        Args:
            env_handle: Handle to the simulation environment.
            camera_props: Backend-specific camera properties (resolution, FOV, clipping planes).

        Returns:
            Integer handle identifying the created camera.
        """
        ...

    def attach_camera_to_body(
        self,
        camera_handle: int,
        env_handle: int,
        body_handle: int,
        local_transform: object,
    ) -> None:
        """Attach a camera to a rigid body so it follows the body's transform.

        Args:
            camera_handle: Handle returned by ``create_camera``.
            env_handle: Handle to the simulation environment.
            body_handle: Handle to the rigid body (actor) the camera tracks.
            local_transform: Pose offset of the camera relative to the body.
        """
        ...

    def render_cameras(self, sim_handle: object) -> None:
        """Trigger rendering for all camera sensors in the simulation.

        Args:
            sim_handle: Handle to the simulation instance.
        """
        ...

    def get_depth_tensor(self, sim_handle: object, env_handle: int, camera_handle: int) -> object:
        """Return a GPU tensor wrapper for the depth image of a single camera.

        The returned object must be convertible to a ``torch.Tensor`` (e.g. via
        ``gymtorch.wrap_tensor`` for Isaac Gym or direct tensor access for Isaac Lab).

        Args:
            sim_handle: Handle to the simulation instance.
            env_handle: Handle to the environment containing the camera.
            camera_handle: Handle identifying the camera.

        Returns:
            A GPU tensor (or wrapper) containing depth pixel values.
        """
        ...

    def get_segmentation_tensor(
        self, sim_handle: object, env_handle: int, camera_handle: int
    ) -> object:
        """Return a GPU tensor wrapper for the segmentation image of a single camera.

        Args:
            sim_handle: Handle to the simulation instance.
            env_handle: Handle to the environment containing the camera.
            camera_handle: Handle identifying the camera.

        Returns:
            A GPU tensor (or wrapper) containing integer segmentation labels.
        """
        ...

    def get_rgb_tensor(self, sim_handle: object, env_handle: int, camera_handle: int) -> object:
        """Return a GPU tensor wrapper for the RGBA image of a single camera.

        Args:
            sim_handle: Handle to the simulation instance.
            env_handle: Handle to the environment containing the camera.
            camera_handle: Handle identifying the camera.

        Returns:
            A GPU tensor (or wrapper) containing RGBA pixel values.
        """
        ...

    def start_tensor_access(self, sim_handle: object) -> None:
        """Begin GPU tensor read access for image data.

        Must be called before reading any image tensors obtained from
        ``get_depth_tensor``, ``get_segmentation_tensor``, or ``get_rgb_tensor``.

        Args:
            sim_handle: Handle to the simulation instance.
        """
        ...

    def end_tensor_access(self, sim_handle: object) -> None:
        """End GPU tensor read access for image data.

        Must be called after finishing reads from image tensors to release
        the GPU synchronization lock.

        Args:
            sim_handle: Handle to the simulation instance.
        """
        ...

    def wrap_tensor(self, raw_tensor: object) -> torch.Tensor:
        """Convert a backend-specific raw GPU tensor into a ``torch.Tensor``.

        Args:
            raw_tensor: The raw tensor object returned by the backend's image API.

        Returns:
            A standard ``torch.Tensor`` backed by the same GPU memory.
        """
        ...
