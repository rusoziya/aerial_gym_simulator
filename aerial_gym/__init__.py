from __future__ import annotations

import os

AERIAL_GYM_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Isaac Gym must be imported before torch (import order requirement).
# Skip when using Isaac Lab backend — isaacgym won't be installed.
_backend = os.environ.get("AERIAL_GYM_BACKEND", "isaacgym").lower()
if _backend != "isaaclab":
    import isaacgym  # noqa: F401

# Import sub-packages to trigger registry auto-registration.
# Each sub-package's __init__.py registers components with the global registries.
import aerial_gym.control  # noqa: F401
import aerial_gym.env_manager  # noqa: F401
import aerial_gym.robots  # noqa: F401
import aerial_gym.sim  # noqa: F401
import aerial_gym.task  # noqa: F401
import aerial_gym.utils  # noqa: F401
