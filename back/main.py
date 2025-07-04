from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
import io
import json
import os
import uuid
from datetime import datetime
import logging
import cv2
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
tokenizer = None
generation_config = None

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for storing images and flashcards
UPLOAD_DIR = Path("uploads")
FLASHCARDS_DIR = Path("flashcards")
UPLOAD_DIR.mkdir(exist_ok=True)
FLASHCARDS_DIR.mkdir(exist_ok=True)

# Initialize model
try:
    model_name  = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.backends.mps.is_available():
        # For Mac, use float32 instead of float16
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None
        )
        model = model.to('mps')    
    

    generation_config = GenerationConfig(
        max_new_tokens=500,
        min_new_tokens=100,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        do_sample=True,
        num_beams=5,
        early_stopping=True,
        length_penalty=1.5,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )
        
    logger.info("model loaded successfully")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise

def save_image(image: Image.Image, filename: str) -> str:
    """Save image to disk and return the file path."""
    try:
        file_path = UPLOAD_DIR / filename
        image.save(file_path)
        return str(file_path)
    except Exception as e:
        logger.error(f"Failed to save image: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save image")

def extract_text_from_image(image: Image.Image) -> str:
    """Extract text from image using Tesseract OCR."""
    try:
        # Check if Tesseract is available
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.error(f"Tesseract not found: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Tesseract OCR is not installed or not found. Please install it from https://github.com/UB-Mannheim/tesseract/wiki"
            )

        def preprocess_image(img):
            img_array = np.array(img)

            # Convert to grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            #resize image
            height, width = img_array.shape
            max_dimension = 2000
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                img_array = cv2.resize(img_array, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            img_array = cv2.bilateralFilter(img_array, 9, 75, 75)            

            # apply adaptive thresholding
            img_array = cv2.adaptiveThreshold(
                img_array, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )          

            return Image.fromarray(img_array)
        
        # preprocess
        img = preprocess_image(image)

        # apply ocr
        custom_config = r'--oem 3 --psm 6 -l eng'
        text = pytesseract.image_to_string(img, config=custom_config)
        text = text.strip()

        if not text:
            logger.warning("No text was extracted from the image")
            raise HTTPException(status_code=400, detail="No text could be extracted from the image. Please ensure the image contains clear, readable text.")
        
        logger.info(f"Successfully extracted {len(text)} characters of text")
        return text
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to extract text from image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")

def generate_summary(text: str) -> str:
    """Generate a summary of the text using Phi-3-mini."""
    try:
        prompt = f"""Create a comprehensive and detailed summary of the following text. 
        Include all key components, their purposes, and relationships.
        Break down complex concepts into clear explanations.
        Ensure the summary is thorough while maintaining clarity.
        
        Text to summarize:
        {text}
        
        Detailed Summary:"""

        # tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        
        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=model.generation_config
            )
        
        # Decode summary
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        summary = clean_summary(summary)

        if "Detailed Summary:" in summary:
            summary = summary.split("Detailed Summary:")[-1].strip()
        
        logger.info(f"Successfully generated summary with {len(summary)} characters")
        return summary
    except Exception as e:
        logger.error(f"Failed to generate summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate summary")


def clean_summary(text: str) -> str:
    """" Clean and format the generated summary. """
    text = text.replace("Create a comprehensive and detailed summary of the following text.", "")
    text = text.replace("Include all key components, their purposes, and relationships.", "")
    text = text.replace("Break down complex concepts into clear explanations.", "")
    text = text.replace("Ensure the summary is thorough while maintaining clarity.", "")     

    # remove response markers
    text = text.replace("### Response:", '').strip()

    # common ocr mistakes
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace("  ", " ")

    # Ensure proper sentence capitalization
    sentences = text.split(". ")
    sentences = [s.capitalize() for s in sentences]
    text = ". ".join(sentences)
    
    # Ensure proper paragraph formatting
    paragraphs = text.split("\n\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    text = "\n\n".join(paragraphs)
    
    return text    

def generate_flashcard_data(image_path: str, original_text: str, summary: str) -> dict:
    """Generate comprehensive flashcard data."""
    try:
        image_size = os.path.getsize(image_path)
        creation_time = datetime.now().isoformat()
        
        flashcard = {
            "id": str(uuid.uuid4()),
            "front": {
                "type": "image",
                "content": str(image_path),
                "metadata": {
                    "size": image_size,
                    "format": "JPEG",
                    "created_at": creation_time
                }
            },
            "back": {
                "type": "text",
                "content": summary,
                "original_text": original_text,
                "metadata": {
                    "source": "BART-large-CNN",
                    "generated_at": creation_time
                }
            },
            "tags": extract_tags(summary),
            "difficulty": calculate_difficulty(summary),
            "created_at": creation_time
        }
        
        # Save flashcard data
        save_flashcard(flashcard)
        return flashcard
    except Exception as e:
        logger.error(f"Failed to generate flashcard data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate flashcard data")

def extract_tags(text: str) -> list:
    """Extract relevant tags from the text."""
    # Simple tag extraction - you might want to use NLP for better results
    words = text.lower().split()
    return list(set([word for word in words if len(word) > 3]))

def calculate_difficulty(text: str) -> str:
    """Calculate difficulty level based on text length and complexity."""
    word_count = len(text.split())
    if word_count < 5:
        return "easy"
    elif word_count < 10:
        return "medium"
    return "hard"

def save_flashcard(flashcard: dict):
    """Save flashcard data to JSON file."""
    try:
        file_path = FLASHCARDS_DIR / f"{flashcard['id']}.json"
        with open(file_path, 'w') as f:
            json.dump(flashcard, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save flashcard: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save flashcard data")

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File must be an image, got {file.content_type}")
        
        # Read and process the uploaded image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
            # Convert to RGB if necessary (for PNG with transparency)
            if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            logger.error(f"Failed to open image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Generate unique filename with original extension
        original_extension = os.path.splitext(file.filename)[1].lower()
        if not original_extension:
            original_extension = '.jpg'
        filename = f"{uuid.uuid4()}{original_extension}"
        
        # Save image
        try:
            image_path = save_image(image, filename)
            logger.info(f"Image saved successfully at {image_path}")
        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save image: {str(e)}")
        
        # Extract text using OCR
        try:
            original_text = extract_text_from_image(image)
            logger.info(f"Text extracted successfully: {len(original_text)} characters")
        except Exception as e:
            logger.error(f"Failed to extract text: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")
        
        if not original_text:
            raise HTTPException(status_code=400, detail="No text found in image")
        
        # Generate summary using T5
        try:
            summary = generate_summary(original_text)
            logger.info(f"Summary generated successfully: {len(summary)} characters")
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")
        
        # Generate and return flashcard
        try:
            flashcard = generate_flashcard_data(image_path, original_text, summary)
            logger.info(f"Flashcard generated successfully with ID: {flashcard['id']}")
            return JSONResponse(content=flashcard)
        except Exception as e:
            logger.error(f"Failed to generate flashcard: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate flashcard: {str(e)}")
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/flashcards")
async def list_flashcards():
    """List all generated flashcards."""
    try:
        flashcards = []
        for file in FLASHCARDS_DIR.glob("*.json"):
            with open(file, 'r') as f:
                flashcards.append(json.load(f))
        return flashcards
    except Exception as e:
        logger.error(f"Failed to list flashcards: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve flashcards")
