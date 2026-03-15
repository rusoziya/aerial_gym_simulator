from __future__ import annotations

import torch

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("IsaacLabCameraBackend")


class IsaacLabCameraBackend:
    """Isaac Lab implementation of the ``CameraBackend`` protocol.

    This is a migration stub. Each method documents the Isaac Gym API call
    it replaces and the Isaac Lab API that should be used instead.
    """

    def __init__(self, simulation: object) -> None:
        """Store a reference to the Isaac Lab simulation.

        Args:
            simulation: The Isaac Lab ``SimulationContext`` or equivalent top-level handle.
        """
        # TODO: Isaac Lab — store ``omni.isaac.lab.sim.SimulationContext`` reference
        # Isaac Gym equivalent: the (gym, sim) handle pair passed around everywhere
        self.simulation = simulation
        self._cameras: dict[tuple[int, int], object] = {}

    def create_camera(self, env_handle: int, camera_props: object) -> int:
        """Create a camera sensor in the given environment.

        Isaac Gym equivalent:
            ``gym.create_camera_sensor(env_handle, camera_props)``
            where ``camera_props`` is a ``gymapi.CameraProperties`` instance with fields:
                enable_tensors, width, height, far_plane, near_plane,
                horizontal_fov, use_collision_geometry

        Isaac Lab replacement:
            Use ``omni.isaac.lab.sensors.CameraCfg`` to declare the camera, then
            instantiate via ``omni.isaac.lab.sensors.Camera(cfg)``.
            Resolution, FOV, and clipping planes are set in ``CameraCfg``.

            Example::

                from omni.isaac.lab.sensors import Camera, CameraCfg

                cfg = CameraCfg(
                    prim_path="/World/envs/env_{env_id}/Robot/camera",
                    update_period=0.0,
                    height=camera_props.height,
                    width=camera_props.width,
                    data_types=["distance_to_image_plane", "semantic_segmentation", "rgb"],
                )
                camera = Camera(cfg)

        Args:
            env_handle: Environment index (used to construct the prim path).
            camera_props: Camera configuration (resolution, FOV, clipping planes).

        Returns:
            An integer handle identifying the created camera.
        """
        # TODO: implement using CameraCfg + Camera(cfg)
        raise NotImplementedError("Isaac Lab camera creation not yet implemented")

    def attach_camera_to_body(
        self,
        camera_handle: int,
        env_handle: int,
        body_handle: int,
        local_transform: object,
    ) -> None:
        """Attach a camera to a rigid body.

        Isaac Gym equivalent:
            ``gym.attach_camera_to_body(camera_handle, env_handle, body_handle,
              local_transform, gymapi.FOLLOW_TRANSFORM)``

        Isaac Lab replacement:
            Attachment is implicit via the USD prim hierarchy. Set the camera's
            ``prim_path`` as a child of the target body prim, e.g.
            ``/World/envs/env_0/Robot/base_link/camera``.
            The offset is specified in ``CameraCfg.offset``:

            Example::

                from omni.isaac.lab.sensors import CameraCfg
                from omni.isaac.lab.utils.math import quat_from_euler_xyz

                cfg = CameraCfg(
                    prim_path=".../base_link/camera",
                    offset=CameraCfg.OffsetCfg(
                        pos=local_transform.p,
                        rot=local_transform.r,
                        convention="world",
                    ),
                    ...
                )

        Args:
            camera_handle: Handle returned by ``create_camera``.
            env_handle: Environment handle / index.
            body_handle: Handle to the rigid body the camera should follow.
            local_transform: Pose offset (position + quaternion).
        """
        # TODO: implement — in Isaac Lab attachment is declarative via prim path
        raise NotImplementedError("Isaac Lab camera attachment not yet implemented")

    def render_cameras(self, sim_handle: object) -> None:
        """Trigger rendering for all camera sensors.

        Isaac Gym equivalent:
            ``gym.render_all_camera_sensors(sim)``

        Isaac Lab replacement:
            Call ``camera.update(dt)`` on each ``Camera`` instance, or rely on
            the ``SimulationContext`` step which triggers sensor updates
            automatically when sensors are registered.

            Example::

                # Option A: explicit update
                for camera in self._cameras.values():
                    camera.update(dt=sim_dt)

                # Option B: automatic via SimulationContext.step()
                # sensors registered with the scene update automatically

        Args:
            sim_handle: Simulation handle (unused in Isaac Lab — kept for interface compat).
        """
        # TODO: implement — call camera.update(dt) or rely on scene step
        raise NotImplementedError("Isaac Lab camera rendering not yet implemented")

    def get_depth_tensor(self, sim_handle: object, env_handle: int, camera_handle: int) -> object:
        """Return the raw depth image tensor for a single camera.

        Isaac Gym equivalent:
            ``gym.get_camera_image_gpu_tensor(sim, env_handle, camera_handle,
              gymapi.IMAGE_DEPTH)``

        Isaac Lab replacement:
            Access ``camera.data.output["distance_to_image_plane"]`` which is
            already a ``torch.Tensor`` on the GPU.

            Example::

                depth = camera.data.output["distance_to_image_plane"]
                # shape: (height, width) — already a torch.Tensor, no wrapping needed

        Args:
            sim_handle: Simulation handle (unused in Isaac Lab).
            env_handle: Environment index.
            camera_handle: Camera handle / index.

        Returns:
            Raw GPU tensor for depth pixels.
        """
        # TODO: implement — return camera.data.output["distance_to_image_plane"]
        raise NotImplementedError("Isaac Lab depth tensor access not yet implemented")

    def get_segmentation_tensor(
        self, sim_handle: object, env_handle: int, camera_handle: int
    ) -> object:
        """Return the raw segmentation image tensor for a single camera.

        Isaac Gym equivalent:
            ``gym.get_camera_image_gpu_tensor(sim, env_handle, camera_handle,
              gymapi.IMAGE_SEGMENTATION)``

        Isaac Lab replacement:
            Access ``camera.data.output["semantic_segmentation"]``.

            Example::

                seg = camera.data.output["semantic_segmentation"]
                # shape: (height, width) — int32 tensor on GPU

        Args:
            sim_handle: Simulation handle (unused in Isaac Lab).
            env_handle: Environment index.
            camera_handle: Camera handle / index.

        Returns:
            Raw GPU tensor for segmentation labels.
        """
        # TODO: implement — return camera.data.output["semantic_segmentation"]
        raise NotImplementedError("Isaac Lab segmentation tensor access not yet implemented")

    def get_rgb_tensor(self, sim_handle: object, env_handle: int, camera_handle: int) -> object:
        """Return the raw RGB(A) image tensor for a single camera.

        Isaac Gym equivalent:
            ``gym.get_camera_image_gpu_tensor(sim, env_handle, camera_handle,
              gymapi.IMAGE_COLOR)``

        Isaac Lab replacement:
            Access ``camera.data.output["rgb"]``.

            Example::

                rgb = camera.data.output["rgb"]
                # shape: (height, width, 4) — RGBA uint8 tensor on GPU

        Args:
            sim_handle: Simulation handle (unused in Isaac Lab).
            env_handle: Environment index.
            camera_handle: Camera handle / index.

        Returns:
            Raw GPU tensor for RGBA pixels.
        """
        # TODO: implement — return camera.data.output["rgb"]
        raise NotImplementedError("Isaac Lab RGB tensor access not yet implemented")

    def start_tensor_access(self, sim_handle: object) -> None:
        """Begin GPU tensor read access for image data.

        Isaac Gym equivalent:
            ``gym.start_access_image_tensors(sim)``

        Isaac Lab replacement:
            Not needed — Isaac Lab camera tensors are standard ``torch.Tensor``
            objects that can be read at any time without explicit locking.

        Args:
            sim_handle: Simulation handle (unused in Isaac Lab).
        """
        # No-op in Isaac Lab: tensors are directly accessible
        pass

    def end_tensor_access(self, sim_handle: object) -> None:
        """End GPU tensor read access for image data.

        Isaac Gym equivalent:
            ``gym.end_access_image_tensors(sim)``

        Isaac Lab replacement:
            Not needed — see ``start_tensor_access``.

        Args:
            sim_handle: Simulation handle (unused in Isaac Lab).
        """
        # No-op in Isaac Lab: tensors are directly accessible
        pass

    def wrap_tensor(self, raw_tensor: object) -> torch.Tensor:
        """Convert a raw GPU tensor into a ``torch.Tensor``.

        Isaac Gym equivalent:
            ``gymtorch.wrap_tensor(raw_tensor)``

        Isaac Lab replacement:
            Isaac Lab already returns ``torch.Tensor`` objects from its camera
            data API, so this is an identity operation.

        Args:
            raw_tensor: The tensor to convert (already a ``torch.Tensor`` in Isaac Lab).

        Returns:
            The same tensor, unchanged.
        """
        if isinstance(raw_tensor, torch.Tensor):
            return raw_tensor
        raise TypeError(f"Expected torch.Tensor from Isaac Lab camera, got {type(raw_tensor)}")
