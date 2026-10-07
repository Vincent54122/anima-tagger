"""Regression checks for prompt validation, without loading the tagger model."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('validator', ROOT/'tools/anima_validate.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ValidationTests(unittest.TestCase):
    def run_cli(self, *args):
        output = io.StringIO()
        with patch.object(sys, 'argv', ['anima_validate', *args, '--json']), contextlib.redirect_stdout(output):
            rc = validator.main()
        return rc, json.loads(output.getvalue())

    def test_rating_removed_and_order_preserved(self):
        rc, result = self.run_cli('--tags', 'general, sunlight, blue_hair, 1girl, questionable')
        self.assertEqual(rc, 0)
        self.assertEqual(result['final_tags'], ['sunlight', 'blue hair', '1girl'])
        self.assertEqual(result['removed_ratings'], ['general', 'questionable'])

    def test_forbidden_tags_reported(self):
        for tag in ['masterpiece', 'best_quality', 'score_8_up', 'BREAK', 'negative prompt']:
            with self.subTest(tag=tag):
                rc, result = self.run_cli('--tags', f'1girl, {tag}')
                self.assertEqual(rc, 1)
                self.assertTrue(result['problems'])

    def test_nl_punctuation_and_final_text(self):
        rc, _ = self.run_cli('--tags', '1girl', '--nl', 'A girl stands outdoors')
        self.assertEqual(rc, 1)
        rc, result = self.run_cli('--tags', '1girl', '--nl', 'A girl stands outdoors.')
        self.assertEqual(rc, 0)
        self.assertEqual(result['final_text'], '1girl,\n\nA girl stands outdoors.')
        self.assertEqual(result['tokens']['count'], validator.count_tokens(result['final_text'])[0])

    def test_wrapped_nl_and_character_parentheses(self):
        rc, _ = self.run_cli('--tags', 'ui (blue archive), 1girl', '--nl', 'A girl stands,\n holding a book.')
        self.assertEqual(rc, 0)

    def test_nl_quotes_and_instructions(self):
        for nl in ['"A girl stands."', 'A girl stands. BREAK', 'Negative Prompt: blur.']:
            with self.subTest(nl=nl):
                rc, _ = self.run_cli('--tags', '1girl', '--nl', nl)
                self.assertEqual(rc, 1)

    def test_mixed_counts_conflict_with_solo(self):
        rc, result = self.run_cli('--tags', '1girl, 1boy, solo, blue hair, red hair')
        self.assertEqual(rc, 1)
        self.assertIn('subject_count', [c['slot'] for c in result['conflicts']])
        self.assertNotIn('hair_color', [c['slot'] for c in result['conflicts']])

    def test_other_and_large_counts_conflict_with_solo(self):
        for tags in ['1girl, 1other, solo', '10girls, solo', '2girls, solo']:
            with self.subTest(tags=tags):
                self.assertEqual(self.run_cli('--tags', tags)[0], 1)

    def test_same_kind_count_conflict(self):
        self.assertEqual(self.run_cli('--tags', '1girl, 2girls')[0], 1)
        self.assertEqual(self.run_cli('--tags', '1girl, 2boys')[0], 0)

    def test_multi_color_tolerance(self):
        self.assertEqual(self.run_cli('--tags', '2girls, blue hair, red hair')[0], 0)
        self.assertEqual(self.run_cli('--tags', '1girl, blue hair, red hair')[0], 1)

    def test_expansion_long_text_and_weight(self):
        nl = 'A girl stands outdoors. ' * 150
        self.assertEqual(self.run_cli('--tags', '1girl', '--nl', nl)[0], 1)
        self.assertEqual(self.run_cli('--tags', '1girl', '--nl', nl, '--no-token-limit')[0], 0)
        self.assertEqual(self.run_cli('--tags', '(blue hair:1.2)', '--no-token-limit')[0], 1)

    def test_tag_layer_overflow_has_actionable_hint(self):
        tags = ', '.join(f'background object number {i}' for i in range(200))
        rc, result = self.run_cli('--tags', tags)
        self.assertEqual(rc, 1)
        self.assertTrue(any('标签层本身超限' in p for p in result['problems']))

    def test_estimate_does_not_pass_budget_check(self):
        with patch.object(validator, 'count_tokens', return_value=(10, 'estimate(rough)')):
            self.assertEqual(self.run_cli('--tags', '1girl')[0], 1)
            self.assertEqual(self.run_cli('--tags', '1girl', '--no-token-limit')[0], 0)

    def test_json_rating_metadata_omitted_and_batch_checks_unk(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'tagger.json'
            path.write_text(json.dumps([{'general': {'1girl': .95, 'solo': .9}, 'rating': {'general': .99}}]), encoding='utf-8')
            rc, result = self.run_cli('--tagger-json', str(path))
            self.assertEqual(rc, 0)
            self.assertEqual(result['final_tags'], ['1girl', 'solo'])
            rc, rows = self.run_cli('--tagger-json', str(path), '--per-record')
            self.assertEqual(rc, 0)
            self.assertEqual(rows[0]['in'], 2)
            path.write_text(json.dumps([{'general': {'中文': .9}}]), encoding='utf-8')
            self.assertEqual(self.run_cli('--tagger-json', str(path), '--per-record')[0], 1)

    def test_nl_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'nl.txt'
            path.write_text('Background:\nA girl stands outdoors.', encoding='utf-8')
            rc, result = self.run_cli('--tags', '1girl', '--nl-file', str(path), '--no-token-limit')
            self.assertEqual(rc, 0)
            self.assertIn('Background:', result['final_text'])


if __name__ == '__main__':
    unittest.main()
