import unittest
from unittest.mock import MagicMock, patch
from backend.routes.upload_handler import upload_item_handler

class TestUploadItemHandler(unittest.TestCase):
    @patch("backend.routes.upload_handler.save_item_to_db")
    @patch("backend.routes.upload_handler.gemini_tag_image")
    @patch("backend.routes.upload_handler.upload_to_firebase")
    @patch("backend.routes.upload_handler.remove_background")
    def test_upload_item_handler(self, mock_remove_bg, mock_upload, mock_gemini, mock_save_db):
        mock_remove_bg.return_value = b'cleaned_image_bytes'
        mock_upload.return_value = 'https://firebase.com/fake.png'
        mock_gemini.return_value = ['shirt', 'blue']
        mock_save_db.return_value = 'item123'

        mock_request = MagicMock()
        mock_request.files.get.return_value = MagicMock(filename="img.png", stream=b"bytes", mimetype="image/png")

        result = upload_item_handler(mock_request, user_id='user1', firebase_uid='abc123')

        self.assertEqual(result['message'], 'Item added successfully')
        self.assertEqual(result['item_id'], 'item123')
