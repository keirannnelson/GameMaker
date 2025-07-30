import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
from backend.services.remove_bg import remove_background

class TestRemoveBackground(unittest.TestCase):
    @patch("backend.services.remove_bg.requests.post")
    def test_remove_background_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"cleaned image"
        mock_post.return_value = mock_response

        mock_file = MagicMock()
        mock_file.filename = "img.jpg"
        mock_file.stream = BytesIO(b"data")
        mock_file.mimetype = "image/jpeg"

        result = remove_background(mock_file)
        self.assertEqual(result, b"cleaned image")

    @patch("backend.services.remove_bg.requests.post")
    def test_remove_background_failure(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_post.return_value = mock_response

        mock_file = MagicMock()
        mock_file.filename = "img.jpg"
        mock_file.stream = BytesIO(b"data")
        mock_file.mimetype = "image/jpeg"

        with self.assertRaises(RuntimeError):
            remove_background(mock_file)
