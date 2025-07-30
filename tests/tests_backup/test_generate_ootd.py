import unittest
from backend.services.gemini import generate_ootd

class TestGenerateOOTD(unittest.TestCase):
    def test_generate_ootd_fallback(self):
        closet = [
            {"id": 1, "type": "shirt", "color": "white", "etiquette": "casual", "description": "white tee"},
            {"id": 2, "type": "jeans", "color": "blue", "etiquette": "casual", "description": "jeans"}
        ]
        result = generate_ootd(["neutral"], closet)
        self.assertEqual(result, [1, 2])
