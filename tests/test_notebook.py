"""Offline checks of the notebook's optional exploration and training path."""

import ast
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

NOTEBOOK = Path(__file__).resolve().parents[1] / "main.ipynb"
CELLS = {cell["id"]: "".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text())["cells"]}


def execute(cell_id, namespace, definitions_only=False):
    tree = ast.parse(CELLS[cell_id])
    if definitions_only:
        tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(tree, f"notebook cell {cell_id}", "exec"), namespace)


class NotebookExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.imports = {"__name__": "__main__"}
        execute("c6fe73f3", cls.imports)

    def setUp(self):
        self.previous_cwd = Path.cwd()
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        os.chdir(self.root)
        self.namespace = self.imports.copy()

    def tearDown(self):
        self.namespace["plt"].close("all")
        os.chdir(self.previous_cwd)
        self.directory.cleanup()

    def test_color_names_absent_skips_both_dependent_cells(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            execute("78cfad93", self.namespace)
            execute("739e7822", self.namespace)
        self.assertIsNone(self.namespace["SC_z1"])
        self.assertIsNone(self.namespace["SC_z2"])
        self.assertIsNone(self.namespace["SC_z3"])
        self.assertIn("Skipping optional Color Names", output.getvalue())
        self.assertIn("GrabCut pipeline continues", output.getvalue())

    def test_color_names_present_runs_exploration_and_fusion(self):
        np = self.namespace["np"]
        self.namespace["scipy"].io.savemat("w2c.mat", {"w2c": np.ones((32768, 11)) / 11})
        for level, size in ((1, 12), (2, 10), (3, 8)):
            self.namespace[f"z{level}_img"] = np.full((size, size, 3), 128, dtype=np.uint8)
            self.namespace[f"SA_z{level}"] = np.zeros((size, size))
            self.namespace[f"SB_z{level}"] = np.zeros((size, size))
        self.namespace["img"] = self.namespace["z1_img"]
        execute("78cfad93", self.namespace)
        execute("739e7822", self.namespace)
        self.assertEqual(self.namespace["saliency_map"].shape, (12, 12))
        self.assertTrue(np.isfinite(self.namespace["saliency_map"]).all())
        self.assertIsNotNone(self.namespace["SC_z1"])

    def test_both_pipelines_complete_one_cpu_epoch_with_real_mobilenet(self):
        from preprocessing import check_csv_done, grabcut_save_from_df

        ns = self.namespace
        np, pd, cv2, torch = (ns[name] for name in ("np", "pd", "cv2", "torch"))
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, old_threads)
        rows = []
        for index in range(4):
            image = np.full((64, 64, 3), 30, dtype=np.uint8)
            cv2.circle(image, (32, 32), 18, (180, 90 + index * 20, 220), -1)
            path = self.root / f"image-{index}.png"
            self.assertTrue(cv2.imwrite(str(path), image))
            rows.append({"image_path": str(path), "label": f"synthetic-{index % 2}", "edible": bool(index % 2)})
        frame = pd.DataFrame(rows)
        with contextlib.redirect_stdout(io.StringIO()):
            for split in ("train", "val", "test"):
                manifest = f"{split}_segmented_paths.csv"
                grabcut_save_from_df(frame, "image_path", manifest)
                self.assertTrue(check_csv_done(frame, manifest))

            execute("b1fa1fdc", ns)
            execute("6bc9b74e", ns)
            execute("0dd80e19", ns, definitions_only=True)
            execute("e78ef3a5", ns, definitions_only=True)
            ns["CONFIG"].NUM_EPOCHS = 1
            # Use the real architecture without downloading pretrained weights.
            constructor = ns["models"].mobilenet_v3_small
            with patch.object(ns["models"], "mobilenet_v3_small", side_effect=lambda **kwargs: constructor(weights=None)):
                for model_type in (1, 2):
                    with self.subTest(model_type=model_type):
                        loader = ns[f"test_loader_{model_type}"]
                        inputs, targets = next(iter(loader))
                        self.assertEqual(tuple(inputs.shape), (4, 3, 224, 224))
                        self.assertEqual(targets.tolist(), [True, False, True, False])
                        model = ns["create_model"]()
                        optimizer = ns["create_optimizer"](model)
                        result = ns["train_experiment"](
                            f"smoke_{model_type}", model,
                            ns[f"train_loader_{model_type}"], ns[f"val_loader_{model_type}"], loader,
                            optimizer, ns["create_scheduler"](optimizer),
                            torch.nn.BCEWithLogitsLoss(), torch.device("cpu"),
                        )
                        self.assertEqual(len(result["history"].train_loss), 1)
                        self.assertTrue(np.isfinite(result["history"].train_loss).all())
                        self.assertEqual(result["test_metrics"]["confusion_matrix"].sum(), 4)
                        self.assertTrue(Path(f"best_model_smoke_{model_type}.pth").is_file())


if __name__ == "__main__":
    unittest.main()
