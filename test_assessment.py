"""Offline checks: python -m unittest -v"""
import io
import json
import unittest

import httpx
from openai import OpenAI
from PIL import Image
from assessment import Assessment, DamagedPart, PANEL_NAMES, assess, panel_summary, prepare_image


def fixture(parts=None):
    return Assessment.model_validate({
        'status': 'Assessable', 'short_description': 'Visible damage to the front bumper.',
        'repair_size': {'category': 'Large', 'confidence': 'Medium',
                        'explanation': 'Severe visible deformation (synthetic test).', 'photo_numbers': [1]},
        'vehicle_type': 'Car', 'powertrain': 'Unknown', 'damaged_parts': parts or [],
        'labour': {'minimum_hours': 2, 'maximum_hours': 4,
                   'basis_and_assumptions': 'Illustrative test estimate only.'},
        'special_needs': [], 'limitations': ['Hidden damage is not visible.'],
        'additional_photos_needed': [],
    })


def part(name, certainty='Observed'):
    return DamagedPart(part=name, damage='Dent', certainty=certainty, photo_numbers=[1])


class AssessmentTests(unittest.TestCase):
    def test_visual_size_independent_of_count(self):
        # One severe panel and many lightly damaged panels must keep the visual label.
        for n, label in [(1, 'Large'), (3, 'Large'), (6, 'Small')]:
            parts = [part(p) for p in PANEL_NAMES[:n]]
            parts += [parts[0], part('Windscreen'), part('Right sill', 'Possible')]
            result = fixture(parts)
            result.repair_size.category = label
            self.assertEqual(panel_summary(result), (n, label))

    def test_unassessable_cannot_have_size(self):
        data = fixture().model_dump()
        data['status'] = 'Insufficient evidence'
        data['labour']['minimum_hours'] = None
        data['labour']['maximum_hours'] = None
        with self.assertRaises(ValueError):
            Assessment.model_validate(data)
        data['repair_size']['category'] = 'Not assessable'
        data['repair_size']['photo_numbers'] = []
        self.assertEqual(panel_summary(Assessment.model_validate(data)), (0, 'Not assessable'))

    def test_bad_labour_range(self):
        data = fixture().model_dump()
        data['labour']['maximum_hours'] = 1
        with self.assertRaises(ValueError):
            Assessment.model_validate(data)

    def test_image_validation(self):
        buf = io.BytesIO()
        Image.new('RGB', (2000, 1000)).save(buf, 'PNG')
        prepared = prepare_image(buf.getvalue())
        with Image.open(io.BytesIO(prepared)) as im:
            self.assertEqual(im.size, (1600, 800))
            self.assertEqual(im.format, 'JPEG')
        with self.assertRaises(ValueError):
            prepare_image(b'not an image')

    def test_responses_sdk_round_trip(self):
        expected = fixture([part('Front bumper')])
        def handler(request):
            self.assertEqual(request.url.path, '/v1/responses')
            body = json.loads(request.content)
            self.assertEqual(body['model'], selected_model)
            if selected_model in {'gpt-5', 'gpt-6-astra'}:
                self.assertEqual(body['reasoning'], {'effort': 'low'})
            self.assertEqual(body['max_output_tokens'], 4000 if selected_model == 'gpt-4.1-mini' else 16000)
            self.assertFalse(body['store'])
            self.assertTrue(body['text']['format']['strict'])
            images = [i for i in body['input'][0]['content'] if i['type'] == 'input_image']
            self.assertEqual(len(images), 2)
            return httpx.Response(200, json={
                'id': 'resp_test', 'object': 'response', 'created_at': 0,
                'status': 'completed', 'model': 'gpt-4.1-mini',
                'output': [{'type': 'message', 'id': 'msg_test', 'status': 'completed',
                            'role': 'assistant', 'content': [{'type': 'output_text',
                            'text': expected.model_dump_json(), 'annotations': []}]}],
            })
        for selected_model in ['gpt-4.1-mini', 'gpt-5-mini', 'gpt-5', 'gpt-6-astra']:
            with OpenAI(api_key='test-only', http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
                result, _ = assess(client, selected_model, [b'image-one', b'image-two'], '')
            self.assertEqual(result, expected)

    def test_refusal(self):
        def handler(request):
            return httpx.Response(200, json={
                'id': 'resp_test', 'object': 'response', 'created_at': 0,
                'status': 'completed', 'model': 'gpt-4.1-mini',
                'output': [{'type': 'message', 'id': 'msg_test', 'status': 'completed',
                            'role': 'assistant', 'content': [{'type': 'refusal', 'refusal': 'Unable to assess.'}]}],
            })
        with OpenAI(api_key='test-only', http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            with self.assertRaises(ValueError):
                assess(client, 'gpt-4.1-mini', [b'image'], '')

if __name__ == '__main__':
    unittest.main()
