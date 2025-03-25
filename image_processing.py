import pytesseract
from PIL import Image
import os

# Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Change path as needed

def extract_text_from_image(image_path):
    """Extracts text from an image using Tesseract OCR"""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"Error processing image: {e}"


# Example usage
if __name__ == "__main__":
    image_text = extract_text_from_image("data/sample_image.jpg")
    print("Extracted Image Text:\n", image_text)
