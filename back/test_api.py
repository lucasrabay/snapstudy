import requests
import os
from pathlib import Path
import json
from datetime import datetime

def test_process_image(image_path: str) -> dict:
    """
    Test the /process-image endpoint with a given image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Response from the API
    """
    url = "http://127.0.0.1:8000/process-image"
    
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Determine MIME type based on file extension
    file_extension = os.path.splitext(image_path)[1].lower()
    mime_type = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp'
    }.get(file_extension, 'image/jpeg')
    
    # Prepare the file
    files = {
        'file': (os.path.basename(image_path), open(image_path, 'rb'), mime_type)
    }
    
    try:
        # Make the request
        print(f"Sending image: {image_path}")
        print(f"Using MIME type: {mime_type}")
        response = requests.post(url, files=files)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Get the response data
        data = response.json()
        
        # Print the results
        print("\nFlashcard created successfully!")
        print("\nFront (Image):")
        print(f"- Path: {data['front']['content']}")
        print(f"- Size: {data['front']['metadata']['size']} bytes")
        
        print("\nBack (Text):")
        print(f"- Content: {data['back']['content']}")
        print(f"- Source: {data['back']['metadata']['source']}")
        
        print("\nMetadata:")
        print(f"- Tags: {', '.join(data['tags'])}")
        print(f"- Difficulty: {data['difficulty']}")
        print(f"- Created at: {data['created_at']}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        raise
    finally:
        # Close the file
        files['file'][1].close()

def test_list_flashcards() -> list:
    """
    Test the /flashcards endpoint to list all flashcards.
    
    Returns:
        list: List of flashcards
    """
    url = "http://127.0.0.1:8000/flashcards"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        flashcards = response.json()
        
        print(f"\nFound {len(flashcards)} flashcards:")
        for card in flashcards:
            print(f"\nID: {card['id']}")
            print(f"Content: {card['back']['content'][:100]}...")
            print(f"Created: {card['created_at']}")
        
        return flashcards
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        raise

if __name__ == "__main__":
    # Create necessary directories
    test_dir = Path("test_images")
    uploads_dir = Path("uploads")
    flashcards_dir = Path("flashcards")
    
    # Create directories if they don't exist
    for directory in [test_dir, uploads_dir, flashcards_dir]:
        directory.mkdir(exist_ok=True)
        print(f"Ensuring directory exists: {directory}")
    
    print("\nAPI Test Script")
    print("==============")
    
    # Test health endpoint
    try:
        response = requests.get("http://127.0.0.1:8000/health")
        response.raise_for_status()
        print("\nServer is healthy!")
    except requests.exceptions.RequestException as e:
        print(f"\nServer is not responding: {str(e)}")
        exit(1)
    
    # Ask for image path
    image_path = input("\nEnter the path to your test image (or press Enter to use a default test image): ").strip()
    
    if not image_path:
        # Use a default test image if available
        default_image = test_dir / "test.jpg"
        if default_image.exists():
            image_path = str(default_image)
        else:
            print("\nNo default test image found. Please provide a path to an image.")
            image_path = input("Enter the path to your test image: ").strip()
    
    # Test the image processing
    try:
        flashcard = test_process_image(image_path)
        
        # Save the response to a JSON file
        output_file = test_dir / f"flashcard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(flashcard, f, indent=2)
        print(f"\nFlashcard data saved to: {output_file}")
        
        # Test listing flashcards
        print("\nTesting /flashcards endpoint...")
        flashcards = test_list_flashcards()
        
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        exit(1) 