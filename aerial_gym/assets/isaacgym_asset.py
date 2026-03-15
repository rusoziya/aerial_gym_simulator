from __future__ import annotations

from isaacgym import gymapi

from aerial_gym.assets.base_asset import BaseAsset


def asset_class_to_AssetOptions(asset_class: object) -> gymapi.AssetOptions:
    """Convert an asset config class to gymapi.AssetOptions."""
    asset_options = gymapi.AssetOptions()
    asset_options.collapse_fixed_joints = asset_class.collapse_fixed_joints
    asset_options.replace_cylinder_with_capsule = asset_class.replace_cylinder_with_capsule
    asset_options.flip_visual_attachments = asset_class.flip_visual_attachments
    asset_options.fix_base_link = asset_class.fix_base_link
    asset_options.density = asset_class.density
    asset_options.angular_damping = asset_class.angular_damping
    asset_options.linear_damping = asset_class.linear_damping
    asset_options.max_angular_velocity = asset_class.max_angular_velocity
    asset_options.max_linear_velocity = asset_class.max_linear_velocity
    asset_options.disable_gravity = asset_class.disable_gravity
    return asset_options


class IsaacGymAsset(BaseAsset):
    def __init__(self, gym, sim, asset_name, asset_file, loading_options):
        super().__init__(asset_name, asset_file, loading_options)
        self.gym = gym
        self.sim = sim
        self.load_from_file(self.file)

    def load_from_file(self, asset_file) -> None:
        file = asset_file.split("/")[-1]
        self.asset = self.gym.load_asset(
            self.sim, self.options.asset_folder, file, self.options.asset_options
        )

        if self.options.place_force_sensor:
            parent_link_idx = self.gym.find_asset_rigid_body_index(
                self.asset, self.options.force_sensor_parent_link
            )
            self.force_sensor_transform = gymapi.Transform()
            self.force_sensor_transform.p = gymapi.Vec3(
                self.options.force_sensor_transform[0],
                self.options.force_sensor_transform[1],
                self.options.force_sensor_transform[2],
            )
            self.force_sensor_transform.r = gymapi.Quat(
                self.options.force_sensor_transform[3],
                self.options.force_sensor_transform[4],
                self.options.force_sensor_transform[5],
                self.options.force_sensor_transform[6],
            )
            sensor_props = gymapi.ForceSensorProperties()
            sensor_props.enable_forward_dynamics_forces = True
            sensor_props.enable_constraint_solver_forces = True
            sensor_props.use_world_frame = False
            self.force_sensor_handle = self.gym.create_asset_force_sensor(
                self.asset, parent_link_idx, self.force_sensor_transform, sensor_props
            )
