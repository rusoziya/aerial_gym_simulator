from __future__ import annotations

import os

import isaacgym

AERIAL_GYM_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Import sub-packages to trigger registry auto-registration.
# Each sub-package's __init__.py registers components with the global registries.
import aerial_gym.control  # noqa: F401
import aerial_gym.env_manager  # noqa: F401
import aerial_gym.robots  # noqa: F401
import aerial_gym.sim  # noqa: F401
import aerial_gym.task  # noqa: F401
import aerial_gym.utils  # noqa: F401
