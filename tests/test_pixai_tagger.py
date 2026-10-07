"""Contract tests for the migration, without loading model weights."""
import argparse
import contextlib
import io
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.pixai_config import CATEGORY_COUNTS, THRESHOLDS, verify_assets
from tools.pixai_tagger import select_tags, require_cuda, main, probability, resolve_device, cpu_preprocess, cuda_available
from tools.anima_validate import load_tagger_json


class PixAITests(unittest.TestCase):
    def fixture(self):
        splits = list(CATEGORY_COUNTS.items())
        tags = [f"{category}_{i}" for category, count in splits for i in range(count)]
        return tags, splits, [0.] * len(tags)

    def test_all_characters_preserved_and_other_categories_excluded_from_validator(self):
        import json
        tags, splits, scores = self.fixture()
        scores[0], scores[1] = .2, .9
        offset = CATEGORY_COUNTS['general']
        scores[offset], scores[offset + 1] = .27, .28
        scores[offset + CATEGORY_COUNTS['character']] = .8
        result = select_tags(scores, tags, splits, THRESHOLDS, top=1)
        self.assertEqual(list(result['general']), ['general_1'])
        self.assertEqual(list(result['character']), ['character_1', 'character_0'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            path.write_text(json.dumps([result]), encoding='utf-8')
            names, confidence = load_tagger_json(str(path))
        self.assertEqual(names, ['character_1', 'character_0', 'general_1'])
        self.assertEqual(confidence['character_0'], .27)
        self.assertNotIn('copyright_0', names)

    def test_bad_output_dimensions_and_nonfinite_scores_rejected(self):
        tags, splits, scores = self.fixture()
        with self.assertRaises(ValueError):
            select_tags(scores[:-1], tags, splits, THRESHOLDS)
        scores[0] = math.nan
        with self.assertRaises(ValueError):
            select_tags(scores, tags, splits, THRESHOLDS)

    def test_cuda_and_bf16_errors_do_not_fallback(self):
        torch = Mock()
        torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'explicitly requested GPU'):
            require_cuda(torch, 'bf16', 'cuda:0')
        torch.cuda.is_available.return_value = True
        torch.cuda.device_count.return_value = 1
        torch.cuda.is_bf16_supported.return_value = False
        with self.assertRaisesRegex(RuntimeError, '--precision fp32'):
            require_cuda(torch, 'bf16', 'cuda:0')
        self.assertEqual(require_cuda(torch, 'fp32', 'cuda:0'), torch.float32)

    def test_missing_assets_and_cli_missing_image(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(len(verify_assets(directory)), 5)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main([str(Path(directory) / 'absent.png'), '--json'])
            self.assertEqual(error.exception.code, 2)

    def test_invalid_thresholds_rejected(self):
        for value in ('nan', 'inf', '-.1', '1.1'):
            with self.assertRaises(argparse.ArgumentTypeError):
                probability(value)

    def test_auto_device_selection_and_explicit_choices(self):
        self.assertEqual(resolve_device('auto', available=False), 'cpu')
        self.assertEqual(resolve_device('auto', available=True), 'cuda:0')
        self.assertEqual(resolve_device('cuda:1', available=False), 'cuda:1')
        self.assertEqual(resolve_device('cpu', available=True), 'cpu')
        with patch.dict(sys.modules, {'torch': None}):
            self.assertFalse(cuda_available())

    def test_cpu_preprocessing_transparency_padding_and_layout(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'transparent.png'
            Image.new('RGBA', (2, 1), (255, 0, 0, 0)).save(path)
            pixels, original_size = cpu_preprocess(path)
        self.assertEqual(original_size, [2, 1])
        self.assertEqual(pixels.shape, (1, 3, 1008, 1008))
        self.assertEqual(pixels.dtype, np.float32)
        self.assertTrue(pixels.flags.c_contiguous)
        np.testing.assert_array_equal(pixels[0, :, 0, 0], [-1, -1, -1])
        np.testing.assert_array_equal(pixels[0, :, 504, 504], [1, 1, 1])

    def test_auto_fallback_emits_clean_json_and_never_loads_gpu_model(self):
        import json
        record = {'device': 'cpu', 'precision': 'fp32', 'performance': {},
                  'general': {'1girl': .9}, 'character': {'candidate': .28}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'exists.png'
            path.touch()
            output, errors = io.StringIO(), io.StringIO()
            with patch('tools.pixai_tagger.cuda_available', return_value=False), \
                 patch('tools.pixai_tagger.PixAITagger') as gpu, \
                 patch('tools.pixai_tagger.PixAICPUTagger') as cpu, \
                 contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                cpu.return_value.tag.return_value = record
                self.assertEqual(main([str(path), '--json']), 0)
                gpu.assert_not_called()
                cpu.assert_called_once()
        result = json.loads(output.getvalue())
        self.assertEqual(result[0]['device'], 'cpu')
        self.assertEqual(result[0]['character'], {'candidate': .28})
        self.assertIn('ONNX CPU FP32', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
