import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from main import app
import os
from PIL import Image, ImageDraw, ImageFont
import io

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_process_image():
    # Create a test image with text
    img = Image.new('RGB', (400, 200), color='white')
    d = ImageDraw.Draw(img)
    
    # Add some text to the image
    text = "Test OCR Text"
    d.text((50, 50), text, fill='black')
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Test image processing
    response = client.post(
        "/process-image",
        files={"file": ("test.png", img_byte_arr, "image/png")}
    )
    
    # Should succeed with text in the image
    assert response.status_code == 200
    assert "id" in response.json()
    assert "front" in response.json()
    assert "back" in response.json()

def test_list_flashcards():
    response = client.get("/flashcards")
    assert response.status_code == 200
    assert isinstance(response.json(), list) 