# image_translator.py
import os
import io
import time
import json
import base64
import zipfile
import requests
import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans
import logging
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from dotenv import load_dotenv
import replicate
from openai import OpenAI
import cv2
from collections import Counter
# Configure logging with elapsed time only
class ElapsedTimeFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = record.created - self.start_time
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = int(elapsed_seconds % 60)
        
        if hours > 0:
            elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            elapsed = f"{minutes:02d}:{seconds:02d}"
            
        return f"{elapsed} - {record.levelname} - {record.getMessage()}"

# Set up root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add new handler with our custom formatter
handler = logging.StreamHandler()
handler.setFormatter(ElapsedTimeFormatter())
logger.addHandler(handler)

class ImageTranslator:
    def __init__(self, input_image_path, target_language='spanish'):
        self.logger = logging.getLogger(__name__)
        self.input_image_path = input_image_path
        self.target_language = target_language.lower()
        load_dotenv()
        self.setup_configs()
        self.font_cache = {}
        
    def setup_configs(self):
        """Initialize configuration and create required directories"""
        # Azure configs
        self.azure_key = os.getenv("AZURE_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT", "https://expense-trend-test.cognitiveservices.azure.com/")
        self.ocr_url = f"{self.azure_endpoint}vision/v3.2/read/analyze"
        self.brand_url = f"{self.azure_endpoint}vision/v3.2/analyze?visualFeatures=Brands"
        
        # WhatFontIs configs
        self.whatfontis_key = os.getenv("WHATFONTIS_API_KEY")
        self.whatfontis_endpoint = "https://www.whatfontis.com/api2/"
        
        # Replicate configs
        self.replicate_token = os.getenv("REPLICATE_API_KEY")
        self.lama_model = "allenhooo/lama:cdac78a1bec5b23c07fd29692fb70baa513ea403a39e643c48ec5edadb15fe72"
        self.segment_model = "tmappdev/lang-segment-anything:891411c38a6ed2d44c004b7b9e44217df7a5b07848f29ddefd2e28bc7cbf93bc"
        
        # Paths and settings
        self.fonts_dir = "fonts"
        self.process_dir = "process-images"
        self.mask_path = os.path.join(self.process_dir, "mask.png")
        self.inpainted_path = os.path.join(self.process_dir, "inpainted.png")
        self.final_path = os.path.join(self.process_dir, "final_image.png")
        
        # Create directories
        os.makedirs(self.fonts_dir, exist_ok=True)
        os.makedirs(self.process_dir, exist_ok=True)
        self.cropped_text_dir = os.path.join(self.process_dir, "cropped-text")
        os.makedirs(self.cropped_text_dir, exist_ok=True)
        
        # OCR/Image settings
        self.padding = 6
        self.blur_radius = 4
        self.confidence_threshold = 0.7  # Text with confidence below 50% will be ignored
        
        # Fallback fonts
        self.fallback_fonts = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/verdana.ttf",
            "arial.ttf"
        ]
        
    def create_segmentation_mask(self):
        """Create detailed text segmentation mask using Lang-SAM"""
        self.logger.info("Creating text segmentation mask")
        try:
            replicate_client = replicate.Client(api_token=self.replicate_token)
            
            # Convert full image to base64
            with open(self.input_image_path, "rb") as f:
                image_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
            
            # Get text mask from Lang-SAM
            output = replicate_client.run(
                self.segment_model,
                input={
                    "image": image_b64,
                    "text_prompt": "text"
                }
            )
            
            # The output could be a URL string, FileOutput object, or other format
            self.logger.info(f"Output type: {type(output)}, content: {output}")
            
            # Handle different output types
            if isinstance(output, str):
                mask_url = output
            elif str(type(output)) == "<class 'replicate.helpers.FileOutput'>":
                # Handle FileOutput type by converting to string
                mask_url = str(output)
            elif hasattr(output, 'output'):
                mask_url = output.output
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                mask_url = str(output[0])
            else:
                self.logger.error(f"Unexpected output format: {output}")
                return None

            if not isinstance(mask_url, str) or not mask_url.startswith('http'):
                self.logger.error(f"Invalid mask URL: {mask_url}")
                return None
                
            self.logger.info(f"Successfully extracted mask URL: {mask_url}")
                
            # Download and process the mask
            self.logger.info(f"Downloading mask from {mask_url}")
            mask_data = requests.get(mask_url).content
            
            # Load the mask and enhance contrast
            mask_image = Image.open(io.BytesIO(mask_data)).convert('L')
            mask_array = np.array(mask_image)
            
            # Enhance contrast: make light grays white and dark grays black
            # Values above 50 will become white (255), below will become black (0)
            enhanced_mask = np.where(mask_array > 50, 255, 0).astype(np.uint8)
            enhanced_mask_image = Image.fromarray(enhanced_mask)
            
            # Save the enhanced mask
            self.segmentation_mask_path = os.path.join(self.process_dir, "text_segmentation.png")
            enhanced_mask_image.save(self.segmentation_mask_path)
            
            self.logger.info(f"Text segmentation mask saved to {self.segmentation_mask_path}")
            return self.segmentation_mask_path
        except Exception as e:
            self.logger.error(f"Failed to create text segmentation mask: {str(e)}")
            return None

    def process_image(self):
        """Main pipeline"""
        try:
            # 1. OCR and Logo Detection
            text_data = self.perform_ocr()
            logo_boxes = self.detect_logos()
            
            # 2. Create detailed text segmentation mask
            self.create_segmentation_mask()
            
            # 3. Create mask for inpainting
            mask_path = self.create_mask(text_data, logo_boxes)
            
            # 4. Font Detection
            text_data = self.detect_fonts(text_data)
            
            # 5. Translation
            text_data = self.translate_text(text_data)
            
            # 5. Inpainting
            inpainted_path = self.perform_inpainting(mask_path)
            
            # 6. Final Image Generation
            final_path = self.generate_final_image(inpainted_path, text_data)
            
            return final_path
        except Exception as e:
            self.logger.error(f"Error in image processing pipeline: {str(e)}")
            raise

    def perform_ocr(self):
        """Extract text using Azure OCR"""
        self.logger.info("Starting OCR process")
        headers = {
            "Ocp-Apim-Subscription-Key": self.azure_key,
            "Content-Type": "application/octet-stream"
        }
        
        with open(self.input_image_path, "rb") as f:
            image_data = f.read()
            
        # Submit OCR request
        response = requests.post(self.ocr_url, headers=headers, data=image_data)
        if response.status_code != 202:
            raise Exception(f"OCR request failed: {response.json()}")
            
        operation_url = response.headers["Operation-Location"]
        
        # Poll for result
        while True:
            result = requests.get(
                operation_url, 
                headers={"Ocp-Apim-Subscription-Key": self.azure_key}
            ).json()
            
            if result["status"] in ["succeeded", "failed"]:
                break
            time.sleep(1)
            
        if result["status"] != "succeeded":
            raise Exception(f"OCR failed: {result}")
            
        lines = result["analyzeResult"]["readResults"][0]["lines"]
        self.logger.info(f"OCR complete: found {len(lines)} text lines")
        return lines
        
    def detect_logos(self):
        """Detect logos using Azure Brand Detection"""
        self.logger.info("Starting logo detection")
        
        with open(self.input_image_path, "rb") as f:
            image_data = f.read()
            
        headers = {
            "Ocp-Apim-Subscription-Key": self.azure_key,
            "Content-Type": "application/octet-stream"
        }
        
        response = requests.post(self.brand_url, headers=headers, data=image_data)
        brand_data = response.json()
        
        logo_boxes = []
        for brand in brand_data.get("brands", []):
            rect = brand["rectangle"]
            logo_boxes.append((
                rect["x"],
                rect["y"],
                rect["x"] + rect["w"],
                rect["y"] + rect["h"]
            ))
            
        self.logger.info(f"Logo detection complete: found {len(logo_boxes)} logos")
        return logo_boxes
        
    def create_mask(self, text_data, logo_boxes):
        """Create mask for inpainting"""
        self.logger.info("Creating mask for inpainting")
        
        image = Image.open(self.input_image_path)
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        
        def overlaps_logo(box, logos):
            x1, y1, x2, y2 = box
            for lx1, ly1, lx2, ly2 in logos:
                if not (x2 < lx1 or x1 > lx2 or y2 < ly1 or y1 > ly2):
                    return True
            return False
            
        for line in text_data:
            text_content = line.get("text", "")
            
            # Check confidence of each word
            word_confidences = [w.get("confidence", 0) for w in line.get("words", [])]
            if not word_confidences:
                self.logger.warning(f"Skipping text '{text_content}' - no confidence data")
                continue
                
            avg_confidence = sum(word_confidences)/len(word_confidences)
            if avg_confidence < self.confidence_threshold:
                self.logger.warning(f"Skipping text '{text_content}' - low confidence ({avg_confidence:.2f})")
                continue
                
            box = line["boundingBox"]
            xs = box[0::2]
            ys = box[1::2]
            
            left = max(min(xs) - self.padding, 0)
            top = max(min(ys) - self.padding, 0)
            right = min(max(xs) + self.padding, image.width)
            bottom = min(max(ys) + self.padding, image.height)
            
            if overlaps_logo((left, top, right, bottom), logo_boxes):
                self.logger.warning(f"Skipping text '{text_content}' - overlaps with logo")
                continue
                
            self.logger.info(f"Adding to mask: '{text_content}' at {left},{top},{right},{bottom}")
            draw.rectangle([left, top, right, bottom], fill=255)
                
        mask = mask.filter(ImageFilter.GaussianBlur(self.blur_radius))
        mask.save(self.mask_path)
        self.logger.info(f"Mask saved to {self.mask_path}")
        return self.mask_path

    def _clear_directory(self, directory):
        """Clear all files in the specified directory"""
        if os.path.exists(directory):
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        self.logger.info(f"Deleted: {file_path}")
                except Exception as e:
                    self.logger.error(f"Error deleting {file_path}: {e}")


    def mode_color(self, pixels):
        """Return the mode of each RGB channel."""
        modes = []
        for i in range(3):
            vals = pixels[:, i]
            counts = Counter(vals)
            mode_val = counts.most_common(1)[0][0]
            modes.append(mode_val)
        return np.array(modes, dtype=int)


    def detect_fonts(self, text_data):
        """Detect and download fonts for each text region, reusing fonts for matching heights."""
        self.logger.info("Starting font detection")
        self._clear_directory(self.cropped_text_dir)
        processed_data = []
        BASE_BRIGHTNESS_MARGIN = 10
        CROP_PADDING = 2

        # Store already identified fonts by rounded height
        height_font_map = {}

        for line in text_data:
            text_content = line["text"]
            box = line["boundingBox"]

            # Crop text region with padding
            image = Image.open(self.input_image_path).convert('RGB')
            xs, ys = box[0::2], box[1::2]
            left = max(min(xs) - CROP_PADDING, 0)
            top = max(min(ys) - CROP_PADDING, 0)
            right = min(max(xs) + CROP_PADDING, image.width)
            bottom = min(max(ys) + CROP_PADDING, image.height)
            cropped = image.crop((left, top, right, bottom))
            cropped_np = np.array(cropped)
            box_height = bottom - top

            # Round height to improve font reuse stability
            box_height_rounded = round(box_height / 2) * 2
            reuse_font_path = height_font_map.get(box_height_rounded)
            if reuse_font_path:
                self.logger.info(f"Reusing font for text '{text_content}' at height {box_height_rounded}px")

            # Analyze text color
            text_info = self._analyze_text_color(box)
            bg_rgb = np.array(text_info["bg_rgb"], dtype=int)

            # Create pseudo-mask for text pixels
            diff = np.linalg.norm(cropped_np - bg_rgb, axis=2)
            text_mask = diff > 30
            text_mask = cv2.dilate(text_mask.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=1).astype(bool)
            text_pixels = cropped_np[text_mask]

            use_clustering = False
            if len(text_pixels) < 10:
                self.logger.info("Thin text detected, using clustering fallback")
                use_clustering = True
                h_c, w_c, _ = cropped_np.shape
                reshaped = cropped_np.reshape(-1, 3)
                kmeans = KMeans(n_clusters=2, random_state=0).fit(reshaped)
                labels = kmeans.labels_
                centers = kmeans.cluster_centers_
                brightness = 0.299*centers[:,0] + 0.587*centers[:,1] + 0.114*centers[:,2]
                cluster_0_count = np.sum(labels == 0)
                cluster_1_count = np.sum(labels == 1)
                text_cluster_idx = 0 if cluster_0_count < cluster_1_count else 1
                text_color = centers[text_cluster_idx]
                text_pixels = reshaped[labels == text_cluster_idx]
            else:
                text_color = self.mode_color(text_pixels)

            text_brightness_vals = 0.299*text_pixels[:,0] + 0.587*text_pixels[:,1] + 0.114*text_pixels[:,2]
            text_brightness = np.mean(text_brightness_vals)
            bg_mask = ~text_mask if not use_clustering else (labels.reshape(cropped_np.shape[:2]) != text_cluster_idx)
            bg_pixels = cropped_np[bg_mask]

            if len(bg_pixels) > 0:
                bg_brightness_vals = 0.299*bg_pixels[:,0] + 0.587*bg_pixels[:,1] + 0.114*bg_pixels[:,2]
                bg_brightness = np.mean(bg_brightness_vals)
            else:
                flat_brightness = 0.299*cropped_np[:,:,0] + 0.587*cropped_np[:,:,1] + 0.114*cropped_np[:,:,2]
                bg_brightness = np.percentile(flat_brightness.flatten(), 95)

            margin = BASE_BRIGHTNESS_MARGIN if cropped_np.size > 500 else 5
            if text_brightness > bg_brightness + margin:
                self.logger.info(f"Inverting text region '{text_content}' - light text on dark background")
                cropped_np = 255 - cropped_np
                text_color = 255 - text_color
                text_info["bg_rgb"] = [255 - c for c in text_info["bg_rgb"]]

            L1, L2 = max(text_brightness, bg_brightness), min(text_brightness, bg_brightness)
            contrast_ratio = (L1 + 0.05) / (L2 + 0.05)
            if contrast_ratio < 4.5:
                self.logger.info(f"Low contrast detected ({contrast_ratio:.2f}) - enhancing contrast")
                lab = cv2.cvtColor(cropped_np, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l_enhanced = clahe.apply(l)
                lab_enhanced = cv2.merge((l_enhanced, a, b))
                cropped_np = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

            cropped = Image.fromarray(cropped_np)
            crop_path = os.path.join(self.cropped_text_dir, f"text_{len(processed_data)}.png")
            cropped.save(crop_path)

            # Identify font if no reuse candidate
            if reuse_font_path:
                font_info = self._get_default_font_info()
                font_info["font_path"] = reuse_font_path
                font_info["font_source"] = "reused"
            else:
                font_info = self.identify_font(crop_path)
                # Cache font by rounded height
                if font_info.get("font_path"):
                    height_font_map[box_height_rounded] = font_info["font_path"]

            font_info.update({
                "text_color": tuple(text_color.astype(int).tolist()) + (255,),
                "text_rgb": text_color.astype(int).tolist(),
                "bg_rgb": text_info["bg_rgb"],
                "text_brightness": text_brightness,
                "bg_brightness": bg_brightness
            })

            word_confidences = [w.get("confidence", 0) for w in line.get("words", [])]
            avg_confidence = sum(word_confidences)/len(word_confidences) if word_confidences else 0

            if avg_confidence >= self.confidence_threshold:
                processed_line = {
                    "id": f"line_{len(processed_data)}",
                    "original": text_content,
                    "boundingBox": box,
                    "confidence": avg_confidence,
                    "translations": {},
                    "font_info": font_info
                }
                self.logger.info(f"Processing text '{text_content}' with confidence {avg_confidence:.2f}")
                processed_data.append(processed_line)
            else:
                self.logger.warning(f"Skipping text '{text_content}' - low confidence ({avg_confidence:.2f})")

        return {"screenshot_id": self.input_image_path, "texts": processed_data}

    def identify_font(self, image_path):
        """Identify font using WhatFontIs API, prioritizing bold/semi-bold fonts.
        Tries all candidates until one successfully downloads."""
        if not os.path.exists(image_path):
            self.logger.error(f"Image not found: {image_path}")
            return self._get_default_font_info()

        try:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "API_KEY": self.whatfontis_key,
                "IMAGEBASE64": 1,
                "urlimagebase64": image_b64,
                "NOTTEXTBOXSDETECTION": 0,
                "FREEFONTS": 1,
                "limit": 20,
            }

            self.logger.info(f"Sending font detection request for {image_path}")
            response = requests.post(self.whatfontis_endpoint, data=payload, timeout=30)

            if response.status_code != 200:
                self.logger.warning(f"Font API returned status {response.status_code}")
                self.logger.warning(f"Response text: {response.text[:500]}")
                return self._get_default_font_info()

            try:
                fonts = response.json()
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON response: {e}")
                self.logger.error(f"Raw response: {response.text}")
                return self._get_default_font_info()

            if not fonts or not isinstance(fonts, list):
                self.logger.warning(f"Font API returned empty or invalid response")
                return self._get_default_font_info()

            self.logger.info(f"Found {len(fonts)} font candidates")

            # Prioritize bold/semi-bold fonts
            bold_keywords = ["bold", "semibold", "semi bold", "demibold", "heavy", "black"]
            def is_boldish(title: str) -> bool:
                if not title:
                    return False
                title_lower = title.lower()
                return any(k in title_lower for k in bold_keywords)

            fonts.sort(key=lambda f: not is_boldish(f.get("title", "")))

            # Try each candidate until one successfully downloads
            for font in fonts:
                font_title = font.get("title", "unknown")
                self.logger.info(f"Trying font: '{font_title}' (is_boldish: {is_boldish(font_title)})")
                font_path = self.download_font(font_title)
                if font_path:
                    self.logger.info(f"Successfully downloaded font: {font_title}")
                    return {
                        "font_title": font_title,
                        "font_url": font.get("url"),
                        "font_path": font_path,
                        "font_source": "ffonts.net",
                        "confidence": font.get("confidence", "unknown")
                    }
                else:
                    self.logger.warning(f"Failed to download font: {font_title}, trying next candidate...")

            # If all candidates fail
            self.logger.warning(f"All font candidates failed, returning default font")
            return self._get_default_font_info()

        except requests.exceptions.Timeout:
            self.logger.error(f"Font detection timed out for {image_path}")
            return self._get_default_font_info()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error during font detection: {e}")
            return self._get_default_font_info()
        except Exception as e:
            self.logger.error(f"Font detection error for {image_path}: {e}", exc_info=True)
            return self._get_default_font_info()
    def _get_default_font_info(self):
        """Return default font info when detection fails"""
        return {
            "font_title": "unknown",
            "font_url": None,
            "font_path": None,
            "font_source": "none",
            "confidence": "unknown"
        }
        
    def _analyze_text_color(self, bounding_box):
        """Analyze text and background colors using segmentation mask or KMeans clustering."""
        try:
            # Load original image
            orig_image = Image.open(self.input_image_path).convert('RGB')
            xs = bounding_box[0::2]
            ys = bounding_box[1::2]
            left, top, right, bottom = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

            # Add padding to capture more context
            padding = 5
            left = max(left - padding, 0)
            top = max(top - padding, 0)
            right = min(right + padding, orig_image.width)
            bottom = min(bottom + padding, orig_image.height)

            # Crop text region
            text_crop = orig_image.crop((left, top, right, bottom))
            text_array = np.array(text_crop)

            # Default return values
            default_result = {
                "color": (0, 0, 0, 255),
                "brightness": 0.0,
                "bg_brightness": 255.0,
                "text_rgb": [0, 0, 0],
                "bg_rgb": [255, 255, 255]
            }

            # Check for segmentation mask
            if hasattr(self, 'segmentation_mask_path') and os.path.exists(self.segmentation_mask_path):
                try:
                    mask_image = Image.open(self.segmentation_mask_path).convert('L')
                    mask_crop = mask_image.crop((left, top, right, bottom))
                    mask_array = np.array(mask_crop)

                    # Adaptive thresholding
                    _, text_mask = cv2.threshold(mask_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    text_mask = text_mask > 0
                    bg_mask = ~text_mask

                    # Extract pixels
                    text_pixels = text_array[text_mask]
                    bg_pixels = text_array[bg_mask]

                    if len(text_pixels) > 10:
                        text_color = np.array(Counter(map(tuple, text_pixels)).most_common(1)[0][0])
                        text_brightness = 0.299 * text_color[0] + 0.587 * text_color[1] + 0.114 * text_color[2]
                    else:
                        self.logger.warning("Insufficient text pixels, falling back to clustering")
                        text_color, text_brightness, bg_color, bg_brightness = self._cluster_colors(text_array)
                        if text_color is None:
                            return default_result

                    if len(bg_pixels) > 10:
                        bg_color = np.median(bg_pixels, axis=0).astype(int)
                        bg_brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                    else:
                        edge_pixels = np.vstack([
                            text_array[0, :], text_array[-1, :],
                            text_array[:, 0], text_array[:, -1]
                        ])
                        bg_color = np.median(edge_pixels, axis=0).astype(int)
                        bg_brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]

                    # Safe contrast adjustment (non-destructive)
                    min_contrast = 4.5
                    contrast_ratio = (max(text_brightness, bg_brightness) + 0.05) / (min(text_brightness, bg_brightness) + 0.05)
                    if contrast_ratio < min_contrast:
                        self.logger.info(f"Low contrast ({contrast_ratio:.2f}), adjusting slightly")
                        if text_brightness > bg_brightness:
                            text_color = np.clip(text_color * 0.9, 0, 255)  # darken slightly
                        else:
                            text_color = np.clip(text_color * 1.1, 0, 255)  # lighten slightly
                        text_brightness = 0.299 * text_color[0] + 0.587 * text_color[1] + 0.114 * text_color[2]

                    text_rgba = tuple(list(text_color.astype(int)) + [255])
                    return {
                        "color": text_rgba,
                        "brightness": float(text_brightness),
                        "bg_brightness": float(bg_brightness),
                        "text_rgb": text_color.astype(int).tolist(),
                        "bg_rgb": bg_color.astype(int).tolist()
                    }

                except Exception as e:
                    self.logger.error(f"Segmentation mask processing failed: {str(e)}")
                    # Fall back to clustering

            # Fallback to KMeans if no mask or fails
            self.logger.info("No valid segmentation mask, using KMeans clustering")
            text_color, text_brightness, bg_color, bg_brightness = self._cluster_colors(text_array)
            if text_color is None:
                return default_result

            text_rgba = tuple(list(text_color.astype(int)) + [255])
            return {
                "color": text_rgba,
                "brightness": float(text_brightness),
                "bg_brightness": float(bg_brightness),
                "text_rgb": text_color.astype(int).tolist(),
                "bg_rgb": bg_color.astype(int).tolist()
            }

        except Exception as e:
            self.logger.error(f"Text color analysis failed: {str(e)}")
            return default_result


    def _cluster_colors(self, image_array):
        """Use KMeans clustering to separate text and background colors."""
        try:
            h, w, _ = image_array.shape
            reshaped = image_array.reshape(-1, 3)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(reshaped)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_.astype(int)

            # Assume text is the cluster with fewer pixels
            cluster_0_count = np.sum(labels == 0)
            cluster_1_count = np.sum(labels == 1)
            text_cluster_idx = 0 if cluster_0_count < cluster_1_count else 1
            bg_cluster_idx = 1 - text_cluster_idx

            text_color = centers[text_cluster_idx]
            bg_color = centers[bg_cluster_idx]

            text_brightness = 0.299 * text_color[0] + 0.587 * text_color[1] + 0.114 * text_color[2]
            bg_brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]

            return text_color, text_brightness, bg_color, bg_brightness
        except Exception as e:
            self.logger.error(f"KMeans clustering failed: {str(e)}")
            return None, None, None, None
    def download_font(self, font_title):
        """
        Download font from ffonts.net or 1001fonts.com if not already cached or saved in the fonts folder.
        Returns the local path to the font file (.ttf).
        """
        import requests
        import os

        # Ensure fonts directory exists
        os.makedirs(self.fonts_dir, exist_ok=True)

        # Safe filename for disk (underscores are fine)
        safe_disk_name = "".join(c if c.isalnum() else "_" for c in font_title)
        font_path_on_disk = os.path.join(self.fonts_dir, f"{safe_disk_name}.ttf")

        # Check in-memory cache
        if font_title in self.font_cache:
            return self.font_cache[font_title]

        # Check fonts folder on disk
        if os.path.exists(font_path_on_disk):
            self.font_cache[font_title] = font_path_on_disk
            return font_path_on_disk

        # Attempt to download from ffonts.net
        safe_ffonts = font_title.replace(" ", "-")
        font_url_ffonts = f"https://www.ffonts.net/{safe_ffonts}.font.download"
        print(f"Trying to download from ffonts.net: {font_url_ffonts}")
        response = requests.get(font_url_ffonts)
        if response.status_code == 200:
            with open(font_path_on_disk, "wb") as f:
                f.write(response.content)
            self.font_cache[font_title] = font_path_on_disk
            return font_path_on_disk

        # Attempt to download from 1001fonts.com (correct dot format)
        safe_1001 = font_title.lower().replace(" ", ".").replace("-", ".")
        font_url_1001 = f"https://st.1001fonts.net/download/font/{safe_1001}.ttf"
        print(f"Trying to download from 1001fonts.com: {font_url_1001}")
        response = requests.get(font_url_1001)
        if response.status_code == 200:
            with open(font_path_on_disk, "wb") as f:
                f.write(response.content)
            self.font_cache[font_title] = font_path_on_disk
            return font_path_on_disk

        raise Exception(f"Failed to download font '{font_title}' from both sources.")
    def extract_font(self, zip_path):
        """Extract font from zip file"""
        try:
            with zipfile.ZipFile(zip_path) as zip_ref:
                font_files = [f for f in zip_ref.namelist() 
                            if f.lower().endswith(('.ttf', '.otf'))]
                
                if not font_files:
                    return None
                    
                font_file = font_files[0]
                zip_ref.extract(font_file, self.fonts_dir)
                return os.path.join(self.fonts_dir, font_file)
                
        except Exception as e:
            self.logger.error(f"Font extraction error: {e}")
            return None

    def translate_text(self, text_data):
        """Translate text using GPT"""
        self.logger.info(f"Starting translation to {self.target_language}")
        client = OpenAI()
        
        prompt = (f"Translate the following English text entries to {self.target_language}. "
                 f"Keep the exact same JSON structure for each entry, but add the translation "
                 f"under translations.{self.target_language}. For example, if the input is "
                 f"{{'id': 'line_1', 'original': 'Hello', 'translations': {{}}}}, "
                 f"output should be {{'id': 'line_1', 'original': 'Hello', 'translations': {{'{self.target_language}': 'TRANSLATION'}}}}. "
                 f"Make sure to translate to {self.target_language} and maintain any formatting or special characters. "
                 f"Return only valid JSON.\n\n")
        prompt += json.dumps(text_data["texts"])
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful translator that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        content = response.choices[0].message.content.strip()
        self.logger.info(f"Raw GPT response: {content}")
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            if content.startswith("json"):
                content = content[4:].strip()
        
        self.logger.info(f"Processed content: {content}")        
        translated_texts = json.loads(content)
        self.logger.info(f"Parsed translations: {json.dumps(translated_texts, indent=2)}")
        
        # Update original data with translations
        for i, entry in enumerate(text_data["texts"]):
            translation = translated_texts[i].get("translations", {}).get(self.target_language)
            if translation:
                entry["translations"][self.target_language] = translation
                self.logger.info(f"Added {self.target_language} translation for '{entry['original']}': '{translation}'")
            else:
                self.logger.warning(f"No translation structure found for text: {entry['original']}")
                
        return text_data

    def perform_inpainting(self, mask_path):
        """Perform inpainting using Replicate"""
        self.logger.info("Starting inpainting")
        replicate_client = replicate.Client(api_token=self.replicate_token)
        
        with open(self.input_image_path, "rb") as f:
            image_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
        with open(mask_path, "rb") as f:
            mask_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
            
        output_url = replicate_client.run(
            self.lama_model,
            input={"image": image_b64, "mask": mask_b64}
        )
        
        inpainted_image_data = requests.get(output_url).content
        with open(self.inpainted_path, "wb") as f:
            f.write(inpainted_image_data)
            
        return self.inpainted_path

    def generate_final_image(self, inpainted_path, text_data):
        """Generate final image with translated text"""
        self.logger.info("Generating final image")
        
        original_image = Image.open(self.input_image_path).convert("RGBA")
        final_image = Image.open(inpainted_path).convert("RGBA")
        draw = ImageDraw.Draw(final_image)
        orig_array = np.array(original_image)
        
        self.logger.info(f"Processing {len(text_data.get('texts', []))} text entries")
        
        for entry in text_data.get("texts", []):
            # We already filtered low confidence in detect_fonts, but double check here
            if entry.get("confidence", 0) < self.confidence_threshold:
                self.logger.warning(f"Skipping text in final image - low confidence: {entry.get('original', 'UNKNOWN')}")
                continue
                
            translated_text = entry["translations"].get(self.target_language, "")
            if not translated_text:
                self.logger.warning(f"No {self.target_language} translation found for text: {entry.get('original', 'UNKNOWN')}")
                continue
                
            box = entry["boundingBox"]
            xs = box[0::2]
            ys = box[1::2]
            left = int(min(xs))
            top = int(min(ys))
            right = int(max(xs))
            bottom = int(max(ys))
            box_width = right - left
            box_height = bottom - top
            
            self.logger.info(f"Processing text: '{translated_text}' in box {left},{top},{right},{bottom}")
            
            # Use the stored text color from font info
            text_color = entry.get('font_info', {}).get('text_color', (0,0,0,255))
            self.logger.info(f"Using stored text color {text_color} for text: {translated_text}")
            
            # Load and scale font
            font_size = int(box_height * 0.8)
            font = None
            
            font_info = entry.get('font_info', {})
            custom_font_path = font_info.get('font_path')
            self.logger.info(f"Attempting to load font: {custom_font_path}")
            
            if custom_font_path and os.path.exists(custom_font_path):
                try:
                    font = ImageFont.truetype(custom_font_path, font_size)
                    self.logger.info(f"Successfully loaded custom font: {custom_font_path}")
                except Exception as e:
                    self.logger.error(f"Failed to load custom font: {str(e)}")
                    font = None
                    
            if font is None:
                self.logger.info("Trying fallback fonts")
                for fallback in self.fallback_fonts:
                    try:
                        font = ImageFont.truetype(fallback, font_size)
                        self.logger.info(f"Successfully loaded fallback font: {fallback}")
                        break
                    except:
                        continue
                        
            if font is None:
                self.logger.warning("Using default font as last resort")
                font = ImageFont.load_default()
                
            # Fit text to box
            bbox = font.getbbox(translated_text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            original_font_size = font_size
            while (text_width > box_width or text_height > box_height) and font_size > 1:
                font_size -= 1
                try:
                    font = ImageFont.truetype(custom_font_path or self.fallback_fonts[0], font_size)
                except:
                    font = ImageFont.load_default()
                bbox = font.getbbox(translated_text)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            
            if original_font_size != font_size:
                self.logger.info(f"Font size adjusted from {original_font_size} to {font_size} to fit box")
            
            # Draw text with enhancements for readability
            x_center = left + box_width / 2
            y_center = top + box_height / 2
            self.logger.info(f"Drawing text '{translated_text}' at position ({x_center}, {y_center})")
            
            # Check if text is white or very light
            r, g, b, a = text_color
            brightness = (r + g + b) / 3
            is_light_text = brightness > 200
            
            if is_light_text:
                # Make white text more opaque
                text_color = (r, g, b, 255)  # Full opacity for white text
                
            
            # Draw main text
            draw.text((x_center, y_center), translated_text,
                     font=font, fill=text_color, anchor="mm")
            
        final_image.save(self.final_path)
        self.logger.info(f"Final image saved with all text rendered")
        return self.final_path


# Example usage:
# translator = ImageTranslator("test-images/test.png", "french")
# final_image_path = translator.process_image()

if __name__ == "__main__":
    # Just modify these two lines to change the image and language
    image_path = "test-images/test5.PNG"
    target_language = "german"
    
    translator = ImageTranslator(image_path, target_language)
    final_image_path = translator.process_image()
    print(f"Translation complete! Final image saved to: {final_image_path}")