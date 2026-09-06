from __future__ import annotations

import sys
import types
import unittest

if sys.platform == "win32":
    resource = types.ModuleType("resource")
    resource.RLIMIT_MEMLOCK = 0
    resource.RLIM_INFINITY = -1
    resource.getrlimit = lambda _kind: (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    resource.setrlimit = lambda _kind, _limits: None
    sys.modules.setdefault("resource", resource)

from verl.trainer.config.config import CheckpointConfig


class PortableCheckpointConfigTest(unittest.TestCase):
    def test_portable_checkpoint_is_opt_in_and_seeded(self) -> None:
        default = CheckpointConfig()
        self.assertFalse(default.portable)
        self.assertEqual(default.portable_seed, 44)

        enabled = CheckpointConfig(portable=True, portable_seed=123)
        self.assertTrue(enabled.portable)
        self.assertEqual(enabled.portable_seed, 123)


if __name__ == "__main__":
    unittest.main()
