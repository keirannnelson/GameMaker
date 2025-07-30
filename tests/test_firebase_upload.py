import unittest
from unittest.mock import MagicMock, patch
from backend.services.firebase_upload import upload_to_firebase

class TestFirebaseUpload(unittest.TestCase):
    @patch("backend.services.firebase_upload.storage.bucket")
    def test_upload_to_firebase(self, mock_bucket):
        mock_blob = MagicMock()
        mock_blob.public_url = "https://firebase.storage.com/fake.png"
        mock_bucket.return_value.blob.return_value = mock_blob

        result = upload_to_firebase(b"image_bytes", "user123", "image/png")
        self.assertTrue(result.startswith("https://firebase.storage.com/"))
