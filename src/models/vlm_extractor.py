"""
VLM-based document text extractor using allenai/olmOCR-2-7B-1025.
Converts PDF page images to text without any API calls.
"""

import os
import torch
from PIL import Image
from typing import List, Optional


_vlm_instance = None  # module-level singleton to avoid reloading


def _get_vlm():
    """Lazy-load the VLM model singleton."""
    global _vlm_instance
    if _vlm_instance is None:
        _vlm_instance = VLMExtractor()
    return _vlm_instance


class VLMExtractor:
    """
    Extracts text from document images using olmOCR-2-7B-1025.

    Usage:
        extractor = VLMExtractor()
        text = extractor.extract_text(pil_image)
    """

    MODEL_ID = "allenai/olmOCR-2-7B-1025"

    def __init__(self, device: Optional[str] = None, max_new_tokens: int = 2048):
        from transformers import AutoProcessor, AutoModelForVision2Seq

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        print(f"[VLMExtractor] Loading {self.MODEL_ID} on {self.device} ...")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        print("[VLMExtractor] Model loaded.")

    def extract_text(self, image: Image.Image, prompt: str = None) -> str:
        """
        Extract text from a PIL Image.

        Args:
            image: PIL Image of the document page.
            prompt: Optional instruction prompt. Defaults to full OCR extraction.

        Returns:
            Extracted text string.
        """
        if prompt is None:
            prompt = (
                "You are an accurate OCR system. "
                "Extract ALL text from this document image exactly as it appears, "
                "preserving structure and layout. Return only the extracted text."
            )

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Decode only newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[0][input_len:]
        text = self.processor.batch_decode([new_ids], skip_special_tokens=True)[0]
        return text.strip()

    def extract_from_pages(self, images: List[Image.Image]) -> List[str]:
        """Extract text from a list of page images."""
        results = []
        for i, img in enumerate(images):
            print(f"[VLMExtractor] Processing page {i + 1}/{len(images)} ...")
            results.append(self.extract_text(img))
        return results


def pdf_to_images(pdf_path: str, dpi: int = 150) -> List[Image.Image]:
    """
    Convert a PDF file to a list of PIL Images (one per page).

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering (default 150 is a good balance of quality/speed).

    Returns:
        List of PIL Image objects.
    """
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=dpi)
        return images
    except ImportError:
        raise ImportError(
            "pdf2image is required for PDF processing. "
            "Install it with: pip install pdf2image"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF to images: {e}")
