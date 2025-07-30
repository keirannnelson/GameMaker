import unittest
from unittest.mock import patch, MagicMock
from backend.services.gemini import gemini_tag_image

class TestGeminiTagImage(unittest.TestCase):
    @patch("backend.services.gemini.requests.get")
    @patch("backend.services.gemini.genai.GenerativeModel")
    def test_gemini_tag_image_json(self, mock_model_class, mock_requests_get):
        mock_requests_get.return_value.content = b"fake_img"
        mock_model = MagicMock()
        mock_model.generate_content.return_value.text = (
            '{ "type": "shirt", "color": "blue", "etiquette": "casual", "description": "cool blue shirt" }'
        )
        mock_model_class.return_value = mock_model
        result = gemini_tag_image("http://img")
        self.assertEqual(result["type"], "shirt")
        self.assertEqual(result["color"], "blue")
        self.assertEqual(result["etiquette"], "casual")
