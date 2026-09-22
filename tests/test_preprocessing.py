import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

import preprocessing


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_source(self, name: str, color=(240, 40, 5)) -> Path:
        path = self.root / name
        image = np.full((64, 64, 3), (10, 20, 30), dtype=np.uint8)
        cv2.circle(image, (32, 32), 14, color, -1)
        self.assertTrue(cv2.imwrite(str(path), image))
        return path

    @staticmethod
    def _frame(paths: list[Path]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "image_path": [str(path) for path in paths],
                "label": [f"label-{index}" for index in range(len(paths))],
                "edible": [index % 2 == 0 for index in range(len(paths))],
            }
        )

    def _generate_cache(
        self,
        df: pd.DataFrame,
        *,
        iters: int = 2,
        cropped_pixels: int = 5,
    ) -> Path:
        manifest = self.root / "segmented.csv"
        preprocessing.grabcut_save_from_df(
            df,
            "image_path",
            str(manifest),
            iters=iters,
            cropped_pixels=cropped_pixels,
        )
        return manifest

    def test_apply_grabcut_keeps_tiny_foreground_fallback(self):
        image = np.full((32, 32, 3), 127, dtype=np.uint8)
        background_mask = np.zeros((32, 32), dtype=np.uint8)

        with patch.object(
            preprocessing.cv2,
            "grabCut",
            return_value=(background_mask, None, None),
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = preprocessing.apply_grabcut(image, cropped_pixels=4)

        self.assertIs(result, image)
        self.assertTrue(
            any("very small foreground" in str(item.message) for item in caught)
        )

    def test_real_generation_preserves_bgr_channels_and_validates_cache(self):
        source = self._write_source("blue-center.png")
        df = self._frame([source])
        manifest = self.root / "segmented.csv"

        results = preprocessing.grabcut_save_from_df(
            df,
            "image_path",
            str(manifest),
            iters=2,
            cropped_pixels=5,
        )

        expected_output = source.with_name("blue-center_grabcut.png")
        self.assertEqual(results, [(str(expected_output), True)])
        generated = cv2.imread(str(expected_output), cv2.IMREAD_COLOR)
        self.assertIsNotNone(generated)
        self.assertEqual(generated[32, 32].tolist(), [240, 40, 5])
        self.assertTrue(
            preprocessing.check_csv_done(
                df,
                str(manifest),
                iters=2,
                cropped_pixels=5,
            )
        )

        saved = pd.read_csv(manifest)
        self.assertEqual(saved["image_path"].tolist(), [str(source)])
        self.assertEqual(saved["grabcut_path"].tolist(), [str(expected_output)])
        self.assertEqual(saved["grabcut_iters"].tolist(), [2])
        self.assertEqual(saved["grabcut_cropped_pixels"].tolist(), [5])

    def test_cache_rejects_missing_and_corrupt_outputs(self):
        source = self._write_source("source.png")
        df = self._frame([source])
        manifest = self._generate_cache(df)
        output = source.with_name("source_grabcut.png")

        output.unlink()
        self.assertFalse(
            preprocessing.check_csv_done(
                df, str(manifest), iters=2, cropped_pixels=5
            )
        )

        output.write_bytes(b"not an image")
        self.assertFalse(
            preprocessing.check_csv_done(
                df, str(manifest), iters=2, cropped_pixels=5
            )
        )

    def test_cache_requires_exact_rows_order_labels_and_expected_paths(self):
        first = self._write_source("first.png")
        second = self._write_source("second.png", color=(5, 40, 240))
        df = self._frame([first, second])
        manifest = self._generate_cache(df)
        valid = pd.read_csv(manifest)

        cases = {}

        superset = pd.concat([valid, valid.iloc[[0]]], ignore_index=True)
        cases["superset"] = superset

        reordered = valid.iloc[::-1].reset_index(drop=True)
        cases["order"] = reordered

        stale_label = valid.copy()
        stale_label.loc[0, "label"] = "stale-label"
        cases["label"] = stale_label

        stale_edible = valid.copy()
        stale_edible.loc[0, "edible"] = not bool(valid.loc[0, "edible"])
        cases["edible"] = stale_edible

        stale_source = valid.copy()
        stale_source.loc[0, "image_path"] = str(self.root / "stale.png")
        cases["source path"] = stale_source

        wrong_output = valid.copy()
        wrong_output.loc[0, "grabcut_path"] = str(first)
        cases["output path"] = wrong_output

        for name, changed in cases.items():
            with self.subTest(name=name):
                changed.to_csv(manifest, index=False)
                self.assertFalse(
                    preprocessing.check_csv_done(
                        df,
                        str(manifest),
                        iters=2,
                        cropped_pixels=5,
                    )
                )
                valid.to_csv(manifest, index=False)

    def test_cache_rejects_requested_subset_and_parameter_changes(self):
        first = self._write_source("first.png")
        second = self._write_source("second.png")
        df = self._frame([first, second])
        manifest = self._generate_cache(df)

        self.assertFalse(
            preprocessing.check_csv_done(
                df.iloc[[0]].reset_index(drop=True),
                str(manifest),
                iters=2,
                cropped_pixels=5,
            )
        )
        self.assertFalse(
            preprocessing.check_csv_done(
                df, str(manifest), iters=3, cropped_pixels=5
            )
        )
        self.assertFalse(
            preprocessing.check_csv_done(
                df, str(manifest), iters=2, cropped_pixels=6
            )
        )

    def test_failed_image_write_raises_and_preserves_previous_cache(self):
        source = self._write_source("source.png")
        df = self._frame([source])
        manifest = self._generate_cache(df)
        output = source.with_name("source_grabcut.png")
        manifest_before = manifest.read_bytes()
        output_before = output.read_bytes()

        with patch.object(preprocessing.cv2, "imwrite", return_value=False):
            with self.assertRaisesRegex(
                preprocessing.PreprocessingError,
                "Failed to process.*source.png.*failed to write segmented image",
            ):
                preprocessing.grabcut_save_from_df(
                    df,
                    "image_path",
                    str(manifest),
                    iters=3,
                    cropped_pixels=5,
                )

        self.assertEqual(manifest.read_bytes(), manifest_before)
        self.assertEqual(output.read_bytes(), output_before)

    def test_publication_failure_invalidates_previous_manifest(self):
        source = self._write_source("source.png")
        df = self._frame([source])
        manifest = self._generate_cache(df)
        output = source.with_name("source_grabcut.png")
        real_replace = preprocessing.os.replace

        def fail_output_publish(source_path, destination_path):
            if Path(destination_path) == output:
                raise OSError("publish failed")
            return real_replace(source_path, destination_path)

        with patch.object(
            preprocessing.os, "replace", side_effect=fail_output_publish
        ):
            with self.assertRaisesRegex(
                preprocessing.PreprocessingError,
                "Failed to publish GrabCut cache.*publish failed",
            ):
                preprocessing.grabcut_save_from_df(
                    df,
                    "image_path",
                    str(manifest),
                    iters=3,
                    cropped_pixels=5,
                )

        self.assertFalse(manifest.exists())
        self.assertFalse(
            preprocessing.check_csv_done(
                df, str(manifest), iters=3, cropped_pixels=5
            )
        )

    def test_segmentation_failure_raises_and_leaves_manifest_unchanged(self):
        source = self._write_source("source.png")
        df = self._frame([source])
        manifest = self._generate_cache(df)
        manifest_before = manifest.read_bytes()

        with patch.object(
            preprocessing,
            "apply_grabcut",
            side_effect=RuntimeError("segmentation exploded"),
        ):
            with self.assertRaisesRegex(
                preprocessing.PreprocessingError,
                "Failed to process.*source.png.*segmentation exploded",
            ):
                preprocessing.grabcut_save_from_df(
                    df,
                    "image_path",
                    str(manifest),
                    iters=3,
                    cropped_pixels=5,
                )

        self.assertEqual(manifest.read_bytes(), manifest_before)

    def test_unreadable_source_raises_without_publishing_manifest(self):
        missing = self.root / "missing.png"
        df = self._frame([missing])
        manifest = self.root / "segmented.csv"

        with self.assertRaisesRegex(
            preprocessing.PreprocessingError,
            "Failed to process.*missing.png.*unable to read source image",
        ):
            preprocessing.grabcut_save_from_df(
                df,
                "image_path",
                str(manifest),
                iters=2,
                cropped_pixels=5,
            )

        self.assertFalse(manifest.exists())


if __name__ == "__main__":
    unittest.main()
