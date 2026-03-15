from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from aerial_gym.task.navigation_task_gate.curriculum_logging import CurriculumLogging

RAD_TO_DEG: float = 57.2958


class CurriculumInfos:
    """Populates task.infos with curriculum metrics for wandb logging.

    Delegates to CurriculumLogging for shared accessor helpers.
    """

    def __init__(self, curriculum_logging: CurriculumLogging) -> None:
        self.cl = curriculum_logging

    @property
    def task(self) -> object:
        return self.cl.task

    def populate(
        self,
        success_rate: float,
        crash_rate: float,
        timeout_rate: float,
        obstacles_behind_gate: int,
        total_obstacles_in_env: int,
    ) -> None:
        """Populate self.task.infos with curriculum metrics for wandb logging."""
        self._populate_basic_metrics(
            success_rate, crash_rate, timeout_rate, obstacles_behind_gate, total_obstacles_in_env
        )
        self._populate_camera_noise_metrics()
        self._populate_frame_dropout_metrics()
        self._populate_camera_angle_metrics()
        self._populate_state_noise_metrics()

    def _populate_basic_metrics(
        self,
        success_rate: float,
        crash_rate: float,
        timeout_rate: float,
        obstacles_behind_gate: int,
        total_obstacles_in_env: int,
    ) -> None:
        infos = self.task.infos
        infos["curriculum/level"] = torch.as_tensor(self.task.curriculum_level, dtype=torch.float32)
        infos["curriculum/progress"] = torch.as_tensor(
            self.task.curriculum_progress_fraction, dtype=torch.float32
        )
        infos["curriculum/success_rate"] = torch.as_tensor(success_rate, dtype=torch.float32)
        infos["curriculum/crash_rate"] = torch.as_tensor(crash_rate, dtype=torch.float32)
        infos["curriculum/timeout_rate"] = torch.as_tensor(timeout_rate, dtype=torch.float32)
        infos["curriculum/obstacles_behind_gate"] = torch.as_tensor(
            obstacles_behind_gate, dtype=torch.float32
        )
        infos["curriculum/total_assets"] = torch.as_tensor(
            total_obstacles_in_env, dtype=torch.float32
        )
        infos["curriculum/max_level_reached"] = torch.as_tensor(
            self.task.max_curriculum_level_reached, dtype=torch.float32
        )

    def _populate_camera_noise_metrics(self) -> None:
        infos = self.task.infos
        (
            gaussian_std,
            dropout_rate,
            eff_drone_std,
            eff_static_std,
            eff_drone_drop,
            eff_static_drop,
        ) = self.cl._get_camera_noise_effective()
        infos["curriculum/camera_gaussian_std"] = torch.as_tensor(gaussian_std, dtype=torch.float32)
        infos["curriculum/camera_dropout_rate"] = torch.as_tensor(dropout_rate, dtype=torch.float32)
        infos["curriculum/camera_noise_drone_gaussian_std"] = torch.tensor(
            eff_drone_std, dtype=torch.float32
        )
        infos["curriculum/camera_noise_static_gaussian_std"] = torch.tensor(
            eff_static_std, dtype=torch.float32
        )
        infos["curriculum/camera_noise_drone_dropout_rate"] = torch.tensor(
            eff_drone_drop, dtype=torch.float32
        )
        infos["curriculum/camera_noise_static_dropout_rate"] = torch.tensor(
            eff_static_drop, dtype=torch.float32
        )

    def _populate_frame_dropout_metrics(self) -> None:
        infos = self.task.infos
        eff, _ = self.cl._get_frame_dropout_effective()
        infos["curriculum/camera_frame_dropout_drone_total"] = torch.tensor(
            eff["drone_total"], dtype=torch.float32
        )
        infos["curriculum/camera_frame_dropout_static_total"] = torch.tensor(
            eff["static_total"], dtype=torch.float32
        )
        infos["curriculum/camera_frame_freeze_drone"] = torch.tensor(
            eff["drone_freeze"], dtype=torch.float32
        )
        infos["curriculum/camera_frame_blank_drone"] = torch.tensor(
            eff["drone_blank"], dtype=torch.float32
        )
        infos["curriculum/camera_frame_freeze_static"] = torch.tensor(
            eff["static_freeze"], dtype=torch.float32
        )
        infos["curriculum/camera_frame_blank_static"] = torch.tensor(
            eff["static_blank"], dtype=torch.float32
        )

    def _populate_camera_angle_metrics(self) -> None:
        infos = self.task.infos
        infos["curriculum/camera_max_angle"] = torch.tensor(
            self.task.max_camera_angle, dtype=torch.float32
        )
        current_angle = self.cl._get_current_camera_angle()
        infos["curriculum/camera_current_angle"] = torch.tensor(current_angle, dtype=torch.float32)
        cam_orient_disabled = bool(
            self.cl._get_gtd().get("static_camera_randomization/orientation_disabled", False)
        )
        infos["curriculum/camera_orientation_randomization_disabled"] = torch.tensor(
            1.0 if cam_orient_disabled else 0.0, dtype=torch.float32
        )

    def _populate_state_noise_metrics(self) -> None:
        if not self.task.task_config.curriculum.enable_state_noise:
            return
        infos = self.task.infos
        sn = self.task.task_config.curriculum.get_state_noise(self.task.curriculum_level)
        infos["curriculum/state_noise_drone_pos_std_m"] = torch.tensor(
            sn["drone_pos_std_m"], dtype=torch.float32
        )
        infos["curriculum/state_noise_drone_orient_std_deg"] = torch.tensor(
            sn["drone_orient_std_rad"] * RAD_TO_DEG, dtype=torch.float32
        )
        infos["curriculum/state_noise_static_pos_std_m"] = torch.tensor(
            sn["static_pos_std_m"], dtype=torch.float32
        )
        infos["curriculum/state_noise_static_orient_std_deg"] = torch.tensor(
            sn["static_orient_std_rad"] * RAD_TO_DEG, dtype=torch.float32
        )
