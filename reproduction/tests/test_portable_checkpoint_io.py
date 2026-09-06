from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from reproduction.portable_checkpoint_backend import (
    load_model_optimizer,
    save_model_optimizer,
)


class PortableCheckpointIOTest(unittest.TestCase):
    def test_dcp_round_trip_restores_model_and_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            torch.manual_seed(7)
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss = model(torch.ones(2, 3)).sum()
            loss.backward()
            optimizer.step()
            expected_weights = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }

            checkpoint = str(Path(temp) / "portable_dcp")
            saved = save_model_optimizer(
                checkpoint, model, optimizer, save_model=True, save_optimizer=True
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            optimizer.state.clear()
            loaded = load_model_optimizer(
                checkpoint, model, optimizer, load_model=True, load_optimizer=True
            )
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected_weights[name]))
            self.assertTrue(optimizer.state)
            self.assertEqual(saved, ["model", "optimizer"])
            self.assertEqual(loaded, saved)
            self.assertTrue((Path(checkpoint) / ".metadata").is_file())


if __name__ == "__main__":
    unittest.main()
