# image_translator.py
import os
import io
import time
import json
import base64
import zipfile
import requests
import numpy as np
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
        # Cache fonts by detected bounding-box height to avoid repeated lookups
        self.height_font_cache = {}
        self.last_api_call = 0
        self.api_call_delay = 1.0  # Delay between API calls in seconds
        # Maximum delay to prevent runaway sleeps when rate limited
        self.api_call_max_delay = 60.0
        # map common language names to ISO codes to handle model output variations
        self.lang_aliases = {
            'german': 'de', 'deutsch': 'de',
            'spanish': 'es', 'espanol': 'es',
            'french': 'fr', 'francais': 'fr',
            'italian': 'it',
            'english': 'en'
        }
        
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
        self.padding = 3
        self.blur_radius = 2
        self.confidence_threshold = 0.83  # Text with confidence below 50% will be ignored
        
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
                        #self.logger.info(f"Deleted: {file_path}")
                except Exception as e:
                    self.logger.error(f"Error deleting {file_path}: {e}")
    # mode_color removed - not used
    def _analyze_all_text_regions(self, text_data):
        """Analyze all text regions at once using the Responses API.

        Returns a dict mapping keys to color info. Keys include the exact OCR text,
        the coordinate key (comma-separated bounding box), and a position key
        (pos_{cx}_{cy}) to allow flexible matching later.
        """

        # Build regions metadata from OCR output
        regions = []
        for line in text_data or []:
            if not isinstance(line, dict):
                continue
            if 'text' not in line or 'boundingBox' not in line:
                continue
            box = line['boundingBox']
            text = line.get('text', '')
            words = line.get('words', [])
            confidence = 0.0
            if words:
                try:
                    confidence = sum(w.get('confidence', 0) for w in words) / len(words)
                except Exception:
                    confidence = 0.0

            left = min(box[0::2])
            top = min(box[1::2])
            right = max(box[0::2])
            bottom = max(box[1::2])
            width = right - left
            height = bottom - top

            regions.append({
                'text': text,
                'box': box,
                'confidence': confidence,
                'bounds': {
                    'left': int(left), 'top': int(top), 'right': int(right), 'bottom': int(bottom),
                    'width': int(width), 'height': int(height)
                }
            })

        # If no regions, nothing to do
        if not regions:
            return {}

        # Build prompt
        prompt = (
            "Given these OCR text regions, determine appropriate text and background colors that meet WCAG AA contrast standards. "
            "Return ONLY valid JSON in this format: {\"regions\": [{\"text\": \"exact OCR text\", \"colors\": {\"text\": [r,g,b], \"background\": [r,g,b]}, \"type\": \"heading|body\", \"contrast_ratio\": number }]}\n\n"
            + json.dumps({'regions': regions}, indent=2)
        )

        # Prepare thumbnail/base64 (keep small)
        try:
            with Image.open(self.input_image_path) as img:
                img = img.convert('RGB')
                img.thumbnail((800, 800), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=80)
                img_bytes = buf.getvalue()
                # Avoid embedding huge images; only embed if small
                if len(img_bytes) <= 200000:  # 200 KB
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    image_payload = f"data:image/jpeg;base64,{img_base64}"
                else:
                    image_payload = None
        except Exception as e:
            self.logger.warning(f"Failed to create thumbnail for GPT analysis: {e}")
            image_payload = None

        client = OpenAI()
        max_retries = 3
        retry_delay = 2
        content_text = None

        for attempt in range(max_retries):
            try:
                # Build input structure for Responses API
                input_item = {
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': prompt}
                    ]
                }
                if image_payload:
                    input_item['content'].append({'type': 'input_image', 'image_url': image_payload})

                resp = client.responses.create(
                    model='gpt-4.1',
                    input=[input_item],
                    temperature=0
                )

                # Extract text from different possible response shapes
                # Prefer `output_text` when available
                if hasattr(resp, 'output_text') and resp.output_text:
                    content_text = resp.output_text
                else:
                    # Try structured output
                    try:
                        out = getattr(resp, 'output', None) or resp.get('output')
                    except Exception:
                        out = None

                    if isinstance(out, list) and out:
                        # collect any text fields
                        parts = []
                        for o in out:
                            if isinstance(o, dict):
                                # some SDKs put content under 'content'
                                c = o.get('content') if isinstance(o.get('content'), (list, dict)) else o.get('text') or o.get('content')
                                if isinstance(c, list):
                                    for item in c:
                                        if isinstance(item, dict) and 'text' in item:
                                            parts.append(item['text'])
                                        elif isinstance(item, str):
                                            parts.append(item)
                                elif isinstance(c, str):
                                    parts.append(c)
                        if parts:
                            content_text = '\n'.join(parts)

                if not content_text:
                    # Fallback: string conversion
                    try:
                        content_text = str(resp)
                    except Exception:
                        content_text = None

                if not content_text:
                    raise ValueError('No text in Responses API output')

                self.logger.info(f"Raw color analysis from GPT: {content_text}")
                break

            except Exception as e:
                self.logger.warning(f"GPT analysis attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                self.logger.error("All GPT analysis attempts failed; falling back to local analysis")
                content_text = None

        if not content_text:
            return {}

        # Strip Markdown code fences if present
        if content_text.strip().startswith('```'):
            try:
                # remove ```json or ```
                content_text = content_text.strip()
                if content_text.startswith('```json'):
                    content_text = content_text.split('```json', 1)[1].rsplit('```', 1)[0]
                else:
                    content_text = content_text.split('```', 2)[1]
            except Exception:
                pass

        # Parse JSON
        try:
            parsed = json.loads(content_text)
        except Exception as e:
            self.logger.error(f"Failed to parse JSON from GPT response: {e}\nResponse was:\n{content_text}")
            return {}

        # Build mapping keyed by text, coordinate string, and position
        result = {}
        for r in parsed.get('regions', []):
            try:
                text = r.get('text', '')
                colors = r.get('colors', {})
                text_rgb = colors.get('text') or colors.get('foreground') or [0, 0, 0]
                bg_rgb = colors.get('background') or [255, 255, 255]
                typ = r.get('type', 'unknown')
                contrast = float(r.get('contrast_ratio', 0)) if r.get('contrast_ratio') is not None else 0.0

                entry = {
                    'text_rgb': [int(x) for x in text_rgb],
                    'bg_rgb': [int(x) for x in bg_rgb],
                    'type': typ,
                    'contrast_ratio': contrast
                }

                # key by exact text (also store a normalized lowercase key so lookups
                # that use .lower() will match - detect_fonts uses a lowercased text key)
                if text:
                    result[text] = entry
                    try:
                        result[text.lower().strip()] = entry
                    except Exception:
                        pass

                # if bounds or box were provided, create coord and pos keys
                # store variants so detect_fonts (which looks for the original
                # boundingBox string and a pos_{cx}_{cy} key) will match reliably
                bounds = r.get('bounds') or r.get('box')
                if bounds:
                    # bounds may be object or list
                    if isinstance(bounds, dict):
                        left = int(bounds.get('left', 0))
                        top = int(bounds.get('top', 0))
                        right = int(bounds.get('right', left))
                        bottom = int(bounds.get('bottom', top))
                    elif isinstance(bounds, list) and len(bounds) >= 4:
                        # assume [left, top, right, bottom]
                        left, top, right, bottom = map(int, bounds[:4])
                    else:
                        left = top = right = bottom = None
                    if left is not None:
                        coord_key = f"{left},{top},{right},{bottom}"
                        result[coord_key] = entry
                        # center pos key
                        cx = int((left + right) / 2)
                        cy = int((top + bottom) / 2)
                        pos_key = f"pos_{cx}_{cy}"
                        result[pos_key] = entry

                # Also, if the original region reported a full box/list (e.g. 8-point
                # bounding box or Azure-style boundingBox), store that exact comma-joined
                # string so detect_fonts' coord_key (which uses the original OCR box
                # joined with commas) will match.
                try:
                    raw_box = r.get('box') or r.get('boundingBox')
                    if isinstance(raw_box, (list, tuple)) and len(raw_box) > 0:
                        raw_coord_key = ','.join(str(int(x)) for x in raw_box)
                        result[raw_coord_key] = entry
                        # also store normalized lowercased text for that coord just in case
                        if text:
                            try:
                                result[f"{raw_coord_key}_text_{text.lower().strip()}"] = entry
                            except Exception:
                                pass
                        # compute center from the raw box points like detect_fonts does
                        try:
                            cx = int(sum(raw_box[0::2]) / len(raw_box[0::2]))
                            cy = int(sum(raw_box[1::2]) / len(raw_box[1::2]))
                            result[f"pos_{cx}_{cy}"] = entry
                        except Exception:
                            pass
                except Exception:
                    pass

            except Exception:
                continue

        self.logger.info(f"Color analysis mapping created with {len(result)} regions")
        return result

    def detect_fonts(self, text_data):
        """Detect and download fonts for each text region with robust handling of thin text."""
        self.logger.info("Starting font detection")
        self._clear_directory(self.cropped_text_dir)
        processed_data = []

        # Get color analysis for all regions at once
        color_analysis = self._analyze_all_text_regions(text_data)
        
        BASE_BRIGHTNESS_MARGIN = 10  # default inversion margin
        CROP_PADDING = 2             # padding around bounding box

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

            # Try multiple ways to match the region with color analysis
            analysis = None
            match_source = None

            # 1. Try exact coordinate match (8-point box joined)
            coord_key = f"{','.join(map(str, box))}"
            if coord_key in color_analysis:
                analysis = color_analysis[coord_key]
                match_source = 'coord'
                self.logger.info(f"Found color match by coordinates for '{text_content}' (key={coord_key})")

            # 2. Try text content match (lowercased)
            if not analysis:
                text_key = text_content.lower().strip()
                if text_key in color_analysis:
                    analysis = color_analysis[text_key]
                    match_source = 'text'
                    self.logger.info(f"Found color match by text content for '{text_content}' (key={text_key})")

            # 3. Try spatial matching (center position)
            if not analysis:
                center_x = sum(box[0::2]) / len(box[0::2])
                center_y = sum(box[1::2]) / len(box[1::2])
                pos_key = f"pos_{int(center_x)}_{int(center_y)}"
                if pos_key in color_analysis:
                    analysis = color_analysis[pos_key]
                    match_source = 'pos'
                    self.logger.info(f"Found color match by position for '{text_content}' (key={pos_key})")
            
            if analysis:
                text_info = {
                    "color": tuple(analysis["text_rgb"] + [255]),
                    "text_rgb": analysis["text_rgb"],
                    "bg_rgb": analysis["bg_rgb"],
                    "brightness": 0.299*analysis["text_rgb"][0] + 0.587*analysis["text_rgb"][1] + 0.114*analysis["text_rgb"][2],
                    "bg_brightness": 0.299*analysis["bg_rgb"][0] + 0.587*analysis["bg_rgb"][1] + 0.114*analysis["bg_rgb"][2],
                    "type": analysis.get("type", "unknown"),
                    "match_source": match_source or 'unknown'
                }
                self.logger.info(f"Color analysis for '{text_content}': text_rgb={analysis['text_rgb']}, bg_rgb={analysis['bg_rgb']}")
            else:
                # Fallback to traditional analysis
                self.logger.warning(f"No color match found for '{text_content}', using local analysis")
                text_info = self._analyze_text_color(box)
                # mark that the color came from local analysis
                text_info["match_source"] = 'local'
            
            bg_rgb = np.array(text_info["bg_rgb"], dtype=int)

            # Rely exclusively on GPT-provided colors for this region.
            # If GPT returned colors for this region, use them directly. If GPT did
            # not provide a color for this region, fall back to conservative defaults
            # (black text on white bg) but DO NOT run local clustering/mode logic.
            used_gpt_color = False
            if analysis and isinstance(analysis.get('text_rgb'), (list, tuple)) and isinstance(analysis.get('bg_rgb'), (list, tuple)):
                # Use GPT colors directly
                text_color = np.array(analysis.get('text_rgb'), dtype=int)
                bg_rgb_arr = np.array(analysis.get('bg_rgb'), dtype=int)
                used_gpt_color = True
                self.logger.info(f"Using GPT-provided colors for '{text_content}': text={text_color.tolist()}, bg={bg_rgb_arr.tolist()}")
                # Build minimal pixel arrays for downstream numeric operations
                text_pixels = np.array([text_color])
                bg_pixels = np.array([bg_rgb_arr])
                text_brightness = 0.299*text_color[0] + 0.587*text_color[1] + 0.114*text_color[2]
                bg_brightness = 0.299*bg_rgb_arr[0] + 0.587*bg_rgb_arr[1] + 0.114*bg_rgb_arr[2]
            else:
                # No GPT analysis for this region — use conservative defaults
                self.logger.warning(f"No GPT color found for '{text_content}'; using default black-on-white (no local clustering).")
                text_color = np.array([0, 0, 0], dtype=int)
                bg_rgb_arr = np.array([255, 255, 255], dtype=int)
                text_pixels = np.array([text_color])
                bg_pixels = np.array([bg_rgb_arr])
                text_brightness = 0.299*text_color[0] + 0.587*text_color[1] + 0.114*text_color[2]
                bg_brightness = 0.299*bg_rgb_arr[0] + 0.587*bg_rgb_arr[1] + 0.114*bg_rgb_arr[2]

            # Adaptive inversion margin
            margin = BASE_BRIGHTNESS_MARGIN if cropped_np.size > 500 else 5
            
            # Invert ONLY if text is brighter than background (light text on dark background)
            if text_brightness > bg_brightness + margin:
                self.logger.info(f"Inverting text region '{text_content}' - light text on dark background (text: {text_brightness:.1f}, bg: {bg_brightness:.1f})")
                # Invert the pixel crop so downstream processing sees dark-on-light,
                # but preserve GPT-provided colors for final drawing when used_gpt_color
                cropped_np = 255 - cropped_np
                if not used_gpt_color:
                    # Only invert stored colors when they were computed locally
                    try:
                        text_color = 255 - text_color
                    except Exception:
                        pass
                    try:
                        text_info["bg_rgb"] = [255 - c for c in text_info.get("bg_rgb", [])]
                    except Exception:
                        pass
                else:
                    # GPT provided the color; keep it as-is for final rendering
                    self.logger.info(f"Preserving GPT color for '{text_content}' despite inversion for processing")
            else:
                self.logger.info(f"No inversion needed for '{text_content}' - text is darker than background (text: {text_brightness:.1f}, bg: {bg_brightness:.1f})")

            # Enhance contrast if low contrast
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

            # Convert back to PIL and save
            cropped = Image.fromarray(cropped_np)
            crop_path = os.path.join(self.cropped_text_dir, f"text_{len(processed_data)}.png")
            cropped.save(crop_path)

            # Detect font (identify_font will try to download/extract a font from the API
            # suggestions; if none is available it will fall back to local name-based lookup)
            # Before calling identify_font (which may call WhatFontIs), check
            # if we've already discovered a font for this integer height and reuse it.
            height_key = int(box[3] - box[1]) if len(box) >= 4 else int(bottom - top)
            font_info = None
            if height_key in self.height_font_cache:
                cached = self.height_font_cache[height_key]
                self.logger.info(f"Reusing cached font for height {height_key}: {cached.get('font_title')} -> {cached.get('font_path')}")
                font_info = cached.copy()
            else:
                font_info = self.identify_font(crop_path)
                # store in height cache if a font_path was found
                try:
                    if font_info and font_info.get('font_path'):
                        self.height_font_cache[height_key] = font_info.copy()
                except Exception:
                    pass
            font_info.update({
                "text_color": tuple(text_color.astype(int).tolist()) + (255,),
                "text_rgb": text_color.astype(int).tolist(),
                "bg_rgb": text_info["bg_rgb"],
                "text_brightness": text_brightness,
                "bg_brightness": bg_brightness
            })
            # no additional fuzzy/visual matches are performed here - identify_font
            # already uses the WhatFontIs suggestion first and then local name lookup
            # Attach provenance so generate_final_image can log where color came from
            try:
                font_info["color_source"] = text_info.get("match_source", match_source or ('gpt' if analysis else 'local'))
                font_info["matched_by"] = match_source
                font_info["used_gpt_color"] = bool(used_gpt_color)
                if used_gpt_color:
                    # preserve the original GPT values for auditing
                    font_info["gpt_text_rgb"] = analysis.get('text_rgb')
                    font_info["gpt_bg_rgb"] = analysis.get('bg_rgb')
            except Exception:
                pass

            # Average OCR confidence
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
        """Identify font using WhatFontIs API with rate limiting"""
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
                "limit": 3
            }
            response = requests.post(self.whatfontis_endpoint, data=payload, timeout=30)
            # Record when the request completed
            self.last_api_call = time.time()

            self.logger.info(f"WhatFontIs response status: {response.status_code}")
            if response.status_code != 200:
                # Log a truncated body for debugging
                try:
                    self.logger.debug(f"WhatFontIs response body: {response.text[:400]}")
                except Exception:
                    pass

            if response.status_code == 429:
                # User requested no rate-limiting logic; just log and return default.
                self.logger.warning("Rate limit exceeded when calling WhatFontIs; returning default font info")
                return self._get_default_font_info()
                
            if response.status_code == 200:
                try:
                    fonts = response.json()
                except Exception:
                    fonts = None
                self.logger.info(f"WhatFontIs returned {len(fonts) if fonts else 0} suggestions")
                if fonts:
                    # For each suggestion, prefer exact local matches first, then try
                    # downloadable assets returned by the API, then try ffonts.net
                    # downloads by name. If none of those succeed, we'll fall back
                    # to a more permissive local similarity search after iterating.
                    name_keys = ['title', 'name', 'font', 'fontname', 'font_title']

                    # 1) Try exact/local substring matches first for each suggestion
                    for font in fonts[:3]:
                        for key in name_keys:
                            name = font.get(key)
                            if not name:
                                continue
                            exact_local = self.find_local_font_exact(name)
                            if exact_local:
                                self.logger.info(f"Successfully matched local font by exact name: {name} (key={key})")
                                return {
                                    "font_title": name,
                                    "font_url": None,
                                    "font_path": exact_local,
                                    "font_source": "local",
                                    "confidence": font.get("confidence", "unknown")
                                }

                    # 2) Try downloadable assets included in the suggestion objects
                    for font in fonts[:3]:  # Try top 3 suggestions
                        try:
                            fetched = self._try_fetch_font_from_suggestion(font)
                            if fetched:
                                self.logger.info(f"Downloaded/extracted font for suggestion: {font.get('title') or font.get('name')}")
                                return {
                                    "font_title": font.get("title", font.get('name', 'unknown')),
                                    "font_url": None,
                                    "font_path": fetched,
                                    "font_source": "whatfontis",
                                    "confidence": font.get("confidence", "unknown")
                                }
                        except Exception as e:
                            self.logger.debug(f"Failed to fetch font from suggestion: {e}")

                    # 3) Try to download from ffonts.net using the suggestion names
                    for font in fonts[:3]:
                        for key in name_keys:
                            name = font.get(key)
                            if not name:
                                continue
                            try:
                                downloaded = self.download_font(name)
                                if downloaded:
                                    self.logger.info(f"Downloaded font from ffonts.net for suggestion name: {name}")
                                    return {
                                        "font_title": name,
                                        "font_url": None,
                                        "font_path": downloaded,
                                        "font_source": "ffonts.net",
                                        "confidence": font.get("confidence", "unknown")
                                    }
                            except Exception as e:
                                self.logger.debug(f"ffonts.net download attempt failed for '{name}': {e}")

                    # 4) Finally, try a permissive local similarity search across suggested names
                    for font in fonts[:3]:
                        for key in name_keys:
                            name = font.get(key)
                            if not name:
                                continue
                            font_path = self.find_local_font(name)
                            if font_path:
                                self.logger.info(f"Successfully matched local font by similar name: {name} (key={key})")
                                return {
                                    "font_title": name,
                                    "font_url": None,
                                    "font_path": font_path,
                                    "font_source": "local",
                                    "confidence": font.get("confidence", "unknown")
                                }

                    try:
                        suggested_titles = [f.get('title') or f.get('name') for f in fonts[:3]]
                    except Exception:
                        suggested_titles = []
                    self.logger.warning(f"No matching local fonts found for suggestions: {suggested_titles}")
            
            self.logger.warning(f"Font detection failed for image '{image_path}': {response.status_code}")
            if response.status_code != 200:
                self.logger.error(f"API response: {response.text}")
            return self._get_default_font_info()
            
        except requests.exceptions.Timeout:
            self.logger.error(f"Font detection timeout for image '{image_path}'")
            return self._get_default_font_info()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Font detection network error for image '{image_path}': {e}")
            return self._get_default_font_info()
        except Exception as e:
            self.logger.error(f"Font detection error for image '{image_path}': {e}")
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

    def _try_fetch_font_from_suggestion(self, font_obj):
        """Attempt to download or extract a font from a WhatFontIs suggestion object.

        Returns a local font file path (ttf/otf) if successful, or None.
        Heuristics: look for keys containing url/zip/file/download; accept http(s) urls or base64 blobs.
        """
        if not isinstance(font_obj, dict):
            return None

        # Candidate keys that may contain downloadable content
        url_keys = ['url', 'download', 'download_url', 'font_url', 'file', 'file_url', 'zip_url']
        bin_keys = ['zip', 'filebase64', 'base64']

        # 1) Try URL-based downloads
        for k in url_keys:
            v = font_obj.get(k)
            if not v or not isinstance(v, str):
                continue
            if v.startswith('http://') or v.startswith('https://'):
                try:
                    r = requests.get(v, timeout=30)
                    if r.status_code == 200:
                        # Determine filename
                        fname = os.path.basename(v.split('?')[0]) or f"font_{int(time.time())}.bin"
                        save_path = os.path.join(self.fonts_dir, fname)
                        with open(save_path, 'wb') as f:
                            f.write(r.content)

                        # If zip, try extract
                        if fname.lower().endswith('.zip'):
                            extracted = self.extract_font(save_path, preferred_title=(font_obj.get('title') or font_obj.get('name') or None))
                            if extracted:
                                return extracted
                        # If ttf/otf, move into fonts dir (already saved) and return
                        if fname.lower().endswith(('.ttf', '.otf')):
                            return save_path
                        # Otherwise, try to inspect content for zip magic
                        if r.content[:4] == b'PK\x03\x04':
                            # write as zip and extract
                            zip_path = save_path if save_path.lower().endswith('.zip') else save_path + '.zip'
                            os.rename(save_path, zip_path)
                            extracted = self.extract_font(zip_path, preferred_title=(font_obj.get('title') or font_obj.get('name') or None))
                            if extracted:
                                return extracted
                except Exception:
                    continue

        # 2) Try base64 blobs
        for k in bin_keys:
            v = font_obj.get(k)
            if not v or not isinstance(v, str):
                continue
            try:
                data = base64.b64decode(v)
                # heuristics: if zip magic present
                if data[:4] == b'PK\x03\x04':
                    zip_name = os.path.join(self.fonts_dir, f"font_{int(time.time())}.zip")
                    with open(zip_name, 'wb') as f:
                        f.write(data)
                    extracted = self.extract_font(zip_name, preferred_title=(font_obj.get('title') or font_obj.get('name') or None))
                    if extracted:
                        return extracted
                # or if ttf/otf magic - try writing and returning
                if data[:2] == b'\x00\x01' or b'trueType' in data[:200].lower() or data[:4] == b'OTTO':
                    fname = os.path.join(self.fonts_dir, f"font_{int(time.time())}.ttf")
                    with open(fname, 'wb') as f:
                        f.write(data)
                    return fname
            except Exception:
                continue

        return None
        
    def _analyze_text_color(self, bounding_box):
        """Analyze text color using clustering without GPT (used as fallback when batch analysis fails)"""
        try:
            image = Image.open(self.input_image_path).convert("RGB")

            # Extract bounding box with padding for context
            xs, ys = bounding_box[0::2], bounding_box[1::2]
            padding = 10  # Add padding to include context
            left = max(0, int(min(xs)) - padding)
            top = max(0, int(min(ys)) - padding)
            right = min(image.width, int(max(xs)) + padding)
            bottom = min(image.height, int(max(ys)) + padding)
            region = image.crop((left, top, right, bottom))

            # Convert to numpy array and cluster
            region_array = np.array(region)
            text_color, bg_color = self._cluster_text_color(region_array)
            text_brightness = 0.299 * text_color[0] + 0.587 * text_color[1] + 0.114 * text_color[2]
            bg_brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
            
            # For low contrast, adjust colors to ensure readability
            contrast_ratio = (max(text_brightness, bg_brightness) + 0.05) / (min(text_brightness, bg_brightness) + 0.05)
            if contrast_ratio < 4.5:  # WCAG AA standard
                self.logger.info(f"Low contrast detected ({contrast_ratio:.2f}), adjusting colors")
                # Make text pure black or white based on background
                if bg_brightness > 127.5:
                    text_color = np.array([0, 0, 0])  # Black text on light background
                else:
                    text_color = np.array([255, 255, 255])  # White text on dark background

            # Calculate brightness
            text_brightness = 0.299 * text_color[0] + 0.587 * text_color[1] + 0.114 * text_color[2]
            bg_brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]

            return {
                "color": tuple(list(text_color) + [255]),
                "brightness": float(text_brightness),
                "bg_brightness": float(bg_brightness),
                "text_rgb": text_color.tolist(),
                "bg_rgb": bg_color.tolist(),
            }

        except Exception as e:
            self.logger.error(f"Text color analysis failed (falling back to black text): {e}")
            return {
                "color": (0, 0, 0, 255),
                "brightness": 0,
                "bg_brightness": 255,
                "text_rgb": [0, 0, 0],
                "bg_rgb": [255, 255, 255],
            }

    def _extract_json_from_text(self, text):
        """Attempt to extract valid JSON from a possibly noisy GPT output.

        Strategies:
        - Strip surrounding code fences (```json ... ``` or ``` ... ```)
        - Find the first '{' and last '}' and attempt to parse progressively smaller substrings
        - Replace single backslashes that commonly break JSON when GPT emits Windows paths
        """
        # quick guard
        if not text or not isinstance(text, str):
            return None

        s = text.strip()
        # Strip code fences
        if s.startswith('```'):
            try:
                # Remove leading fence and trailing fence
                s = s.split('```', 2)[1]
                # If starts with json label, remove it
                if s.strip().lower().startswith('json'):
                    s = s.strip()[4:].strip()
            except Exception:
                pass

        # Fix common escaping: replace backslashes in Windows-style paths with forward slashes
        s = s.replace('\\', '/')

        # Find the first JSON-like object or array in the text
        first_obj = s.find('{')
        first_arr = s.find('[')
        start = -1
        if first_obj != -1 and (first_arr == -1 or first_obj < first_arr):
            start = first_obj
            open_char, close_char = '{', '}'
        elif first_arr != -1:
            start = first_arr
            open_char, close_char = '[', ']'
        else:
            # No obvious JSON; try parse entire string
            try:
                return json.loads(s)
            except Exception:
                self.logger.error('Failed to locate JSON in GPT response')
                return None

        # Try progressively trimming trailing content until JSON parses or we fail
        for end in range(len(s), start, -1):
            if s[end-1] != close_char:
                continue
            candidate = s[start:end]
            try:
                return json.loads(candidate)
            except Exception:
                continue

        # As a last resort, try to repair some common problems and parse
        candidate = s[start:]
        # Attempt naive balancing of braces/brackets by counting
        if open_char == '{':
            open_count = candidate.count('{')
            close_count = candidate.count('}')
            if close_count < open_count:
                # append missing braces
                candidate = candidate + ('}' * (open_count - close_count))
        else:
            open_count = candidate.count('[')
            close_count = candidate.count(']')
            if close_count < open_count:
                candidate = candidate + (']' * (open_count - close_count))

        try:
            return json.loads(candidate)
        except Exception as e:
            self.logger.error(f"Final JSON extraction attempt failed: {e}")
            return None


    def _cluster_text_color(self, region):
        """Cluster pixels to identify text vs background."""
        pixels = region.reshape(-1, 3).astype(float)
        if len(pixels) > 10000:  # downsample for speed
            idx = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[idx]

        kmeans = KMeans(n_clusters=2, n_init=5, random_state=0)
        kmeans.fit(pixels)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = np.bincount(labels)

        # Pick the smaller cluster as text (assumes text is minority)
        if counts[0] < counts[1]:
            text_cluster, bg_cluster = 0, 1
        else:
            text_cluster, bg_cluster = 1, 0

        text_color = centers[text_cluster]
        bg_color = centers[bg_cluster]

        # DO NOT INVERT HERE - return actual detected colors
        return text_color, bg_color

    def find_local_font(self, font_title):
        """Find a font in the local fonts directory using fuzzy matching"""
        if not font_title:
            return None
            
        if font_title in self.font_cache:
            self.logger.info(f"Using cached font: {font_title} -> {self.font_cache[font_title]}")
            return self.font_cache[font_title]
            
        try:
            # Break down font title into components for better matching
            font_terms = font_title.lower().split()
            base_font_name = font_terms[0] if font_terms else ""
            weight_hints = {'bold', 'light', 'medium', 'heavy', 'black', 'thin', 'regular', 'semibold', 'italic'}
            style_terms = {term for term in font_terms if term in weight_hints}
            
            # First try exact matches
            for file in os.listdir(self.fonts_dir):
                if not file.lower().endswith(('.ttf', '.otf')):
                    continue
                    
                file_name = os.path.splitext(file)[0].lower()
                if font_title.lower() in file_name:
                    font_path = os.path.join(self.fonts_dir, file)
                    try:
                        ImageFont.truetype(font_path, 12)
                        self.logger.info(f"Found exact matching font: {file}")
                        self.font_cache[font_title] = font_path
                        return font_path
                    except Exception:
                        continue
            
            # Then try partial matches
            best_match = None
            best_match_score = 0
            
            for file in os.listdir(self.fonts_dir):
                if not file.lower().endswith(('.ttf', '.otf')):
                    continue
                    
                file_name = os.path.splitext(file)[0].lower()
                file_terms = set(file_name.split())
                
                # Calculate match score
                score = 0
                if base_font_name in file_name:
                    score += 3  # High priority for base font name match
                score += len(style_terms & file_terms)  # Add points for matching style terms
                
                if score > best_match_score:
                    font_path = os.path.join(self.fonts_dir, file)
                    try:
                        ImageFont.truetype(font_path, 12)
                        best_match = font_path
                        best_match_score = score
                    except Exception:
                        continue
            
            if best_match:
                self.logger.info(f"Found best matching font: {os.path.basename(best_match)} (score: {best_match_score})")
                self.font_cache[font_title] = best_match
                return best_match
                            
            self.logger.warning(f"No matching local font found for: {font_title}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error searching for local font {font_title}: {e}")
            return None

    def find_local_font_exact(self, font_title):
        """Return exact/local substring match for a font title in the local fonts directory.

        This is stricter than find_local_font: it tries to find a filename that
        contains the suggestion title as a contiguous substring (case-insensitive).
        """
        if not font_title:
            return None

        try:
            target = font_title.lower()
            for file in os.listdir(self.fonts_dir):
                if not file.lower().endswith(('.ttf', '.otf')):
                    continue
                file_name = os.path.splitext(file)[0].lower()
                if target in file_name:
                    path = os.path.join(self.fonts_dir, file)
                    try:
                        ImageFont.truetype(path, 12)
                        return path
                    except Exception:
                        continue
            return None
        except Exception as e:
            self.logger.error(f"Error during exact local font lookup for '{font_title}': {e}")
            return None

    def download_font(self, font_title):
        """Download font from ffonts.net by inferred safe title and extract it.

        Returns the path to the extracted font file if successful, or None.
        Uses a simple cache to avoid repeated downloads during a run.
        """
        if not font_title:
            return None

        if font_title in self.font_cache:
            return self.font_cache[font_title]

        try:
            # sanitize title into a safe ffonts.net slug
            safe_title = "".join(c for c in font_title.replace(' ', '-') 
                                   if c.isalnum() or c in ('-', '_')).rstrip()

            download_url = f"https://www.ffonts.net/{safe_title}.font.zip"
            self.logger.info(f"Attempting to download font from ffonts.net: {download_url}")
            response = requests.get(download_url, timeout=30)

            # Basic check for zip magic
            if not response.content or not response.content.startswith(b'PK'):
                self.logger.info(f"ffonts.net did not return a zip for '{font_title}' (url: {download_url})")
                return None

            zip_path = os.path.join(self.fonts_dir, f"{safe_title}.zip")
            with open(zip_path, 'wb') as f:
                f.write(response.content)

            font_path = self.extract_font(zip_path, preferred_title=font_title)
            try:
                os.remove(zip_path)
            except Exception:
                pass

            if font_path:
                self.logger.info(f"Successfully downloaded and extracted font '{font_title}' -> {font_path}")
                self.font_cache[font_title] = font_path
                return font_path
            else:
                self.logger.info(f"No font files found inside downloaded zip for '{font_title}'")
                return None

        except Exception as e:
            self.logger.error(f"Font download error for '{font_title}': {e}")
            return None

    def extract_font(self, zip_path, preferred_title=None):
        """Extract font from zip file.

        Heuristics:
        - Prefer filenames that match common weight names (regular, medium, bold, semibold)
        - Avoid decorative variants (outline, shadow, stencil, demo, trial)
        - If preferred_title is provided, boost files that contain parts of that title
        """
        try:
            with zipfile.ZipFile(zip_path) as zip_ref:
                font_files = [f for f in zip_ref.namelist()
                              if f.lower().endswith(('.ttf', '.otf'))]

                if not font_files:
                    return None

                # Ranking heuristic
                def score_name(name: str) -> int:
                    n = name.lower()
                    score = 0
                    # prefer common weights; these are baseline boosts
                    # Note: biasing toward bold/black/extra-bold by default when no preferred_title
                    if any(k in n for k in ['regular', 'roman', 'normal', 'book']):
                        score += 30
                    if any(k in n for k in ['medium', 'med']):
                        score += 15
                    # semibold is less preferred than true bold/black
                    if any(k in n for k in ['semibold', 'semi-bold', 'semib']):
                        score += 10
                    # strong boost for bold-like weights
                    if any(k in n for k in ['black', 'extra-bold', 'extrabold', 'heavy']):
                        score += 120
                        if not preferred_title:
                            # extra bias when user didn't request a specific title
                            score += 60
                    elif 'bold' in n:
                        # generic 'bold' (catch other bold variants) gets a strong boost too
                        score += 80
                        if not preferred_title:
                            score += 40
                    # penalize decorative/outline/demo/trial variants
                    if any(k in n for k in ['outline', 'shadow', 'stencil', 'stamp', 'demo', 'trial', 'oblique', 'inline']):
                        score -= 50
                    # prefer ttf slightly over otf (arbitrary)
                    if n.endswith('.ttf'):
                        score += 2
                    # boost if parts of the preferred title are present
                    if preferred_title:
                        for part in preferred_title.lower().replace('-', ' ').split():
                            if part and part in n:
                                score += 10
                    return score

                # Score all candidates and log them for transparency
                scored = [(f, score_name(f)) for f in font_files]
                scored.sort(key=lambda x: x[1], reverse=True)

                # Log all candidates and scores (useful for debugging selection)
                try:
                    self.logger.info("Font candidates inside zip (name -> score):\n" + "\n".join([f"  {s[0]} -> {s[1]}" for s in scored]))
                except Exception:
                    pass

                best_file, best_score = scored[0]
                # If best_file isn't the first listed in the archive, log that too
                if best_file != font_files[0]:
                    self.logger.info(f"Chose best font in zip: '{best_file}' (score={best_score}) over first: '{font_files[0]}'")

                # Ensure extraction path exists
                extract_dir = self.fonts_dir
                zip_ref.extract(best_file, extract_dir)
                extracted_path = os.path.join(extract_dir, best_file)
                # Normalize path separators for Windows
                extracted_path = os.path.normpath(extracted_path)
                return extracted_path

        except Exception as e:
            self.logger.error(f"Font extraction error: {e}")
            return None

    def translate_text(self, text_data):
        """Translate text using GPT"""
        self.logger.info(f"Starting translation to {self.target_language}")
        client = OpenAI()
        
        prompt = (
            f"Translate the following English text entries to {self.target_language}. "
            "Keep the exact same JSON structure for each entry, but add the translation "
            f"under translations.{self.target_language}.\n\n"
            "IMPORTANT: Preserve the ORIGINAL CAPITALIZATION PATTERN of each entry. "
            "Match the casing style of the original when producing the translation: "
            "e.g. 'HELLO' -> 'AUREVOIR' (all-caps), 'Hello' -> 'Au revoir' (capitalized), "
            "'hello' -> 'au revoir' (lowercase), 'Hello World' -> 'Au revoir Monde' (Title Case). "
            "If the original uses mixed-case per-word, apply the same per-word casing pattern to each translated word. "
            "Preserve punctuation and special characters; only translate the words themselves. "
            f"Return only valid JSON with the same array structure as the input.\n\n"
        )
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

        # Try bulk parse first
        parsed = self._extract_json_from_text(content)
        if parsed is None or not isinstance(parsed, list) or len(parsed) != len(text_data.get("texts", [])):
            # Bulk parse failed or returned unexpected shape — attempt batched fallback translations
            # NOTE: We avoid per-line calls to reduce cost. The batched fallback will request translations
            # in groups and will ONLY accept results that include the exact language name as the key
            # (e.g. 'german'). Do NOT accept ISO codes like 'de'. If a batch fails to return the exact
            # key, we will fallback to using the original text as the translation for that batch.
            self.logger.warning("Bulk translation JSON parse failed or mismatched size; attempting batched fallback (no per-line calls)")
            texts = text_data.get("texts", [])
            batch_size = 12
            parsed = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                originals = [entry.get('original', '') for entry in batch]

                batch_prompt = (
                    f"Translate the following list of English text entries to {self.target_language}.\n"
                    "Return ONLY valid JSON: an array of objects, each object corresponding to the input order.\n"
                    "Each object MUST contain a 'translations' object with a single key that is EXACTLY the target language NAME (not an ISO code).\n"
                    f"Each object MUST use the key '{self.target_language}' inside 'translations' (do NOT use 'de' or any ISO code).\n"
                    "DO NOT include any prose, headers, or a line like 'Target language: ...' before the JSON array.\n"
                    "Preserve the ORIGINAL CAPITALIZATION PATTERN of each entry (keep all-caps, title-case, lowercase, etc.).\n\n"
                    "Input array (same order should be preserved):\n"
                )
                batch_prompt += json.dumps(originals, ensure_ascii=False)

                try:
                    batch_resp = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": f"You are a helpful translator that outputs JSON arrays matching the input order. The target language is '{self.target_language}'. Only output the JSON array and nothing else."},
                            {"role": "user", "content": batch_prompt}
                        ],
                        temperature=0
                    )
                    batch_content = batch_resp.choices[0].message.content.strip()
                    self.logger.info(f"Raw batch response (items={len(batch)}): {batch_content[:1000]}")
                    batch_parsed = self._extract_json_from_text(batch_content)

                    # Validate parsed shape: must be a list with same length and each item must include
                    # translations.<exact language name>
                    valid = True
                    if not isinstance(batch_parsed, list) or len(batch_parsed) != len(batch):
                        valid = False
                    else:
                        for obj in batch_parsed:
                            if not isinstance(obj, dict):
                                valid = False
                                break
                            translations = obj.get('translations')
                            if not isinstance(translations, dict) or self.target_language not in translations:
                                valid = False
                                break

                    if valid:
                        parsed.extend(batch_parsed)
                    else:
                        # Try one recovery attempt with a stricter prompt asking for ONLY the JSON array and nothing else.
                        self.logger.warning(f"Batch translation did not return required exact-key translations for target '{self.target_language}'. Attempting one strict retry before falling back.")
                        strict_prompt = (
                            "RETURN ONLY a JSON array (no prose, no comments, no headers). Each array item must correspond to the input order.\n"
                            f"Each item MUST be an object containing 'translations' with a single key EXACTLY named '{self.target_language}'.\n"
                            "Do NOT print the target language name or any extra text outside the JSON array.\n\n"
                        )
                        strict_prompt += json.dumps(originals, ensure_ascii=False)
                        try:
                            retry_resp = client.chat.completions.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": f"You are a JSON-only translator. The target language is '{self.target_language}'. Output strictly valid JSON arrays matching the input order and do not print any prose or headers."},
                                    {"role": "user", "content": strict_prompt}
                                ],
                                temperature=0
                            )
                            retry_content = retry_resp.choices[0].message.content.strip()
                            self.logger.info(f"Raw strict-retry batch response: {retry_content[:1000]}")
                            retry_parsed = self._extract_json_from_text(retry_content)

                            # Re-validate retry result
                            retry_valid = True
                            if not isinstance(retry_parsed, list) or len(retry_parsed) != len(batch):
                                retry_valid = False
                            else:
                                for obj in retry_parsed:
                                    if not isinstance(obj, dict):
                                        retry_valid = False
                                        break
                                    translations = obj.get('translations')
                                    if not isinstance(translations, dict) or self.target_language not in translations:
                                        retry_valid = False
                                        break

                            if retry_valid:
                                parsed.extend(retry_parsed)
                            else:
                                self.logger.warning(f"Strict retry still did not return required key '{self.target_language}'. Falling back to original text for this batch.")
                                for src in originals:
                                    parsed.append({"translations": {self.target_language: src}})
                        except Exception as e:
                            self.logger.error(f"Strict retry failed for items {i}-{i+len(batch)-1}: {e}")
                            for src in originals:
                                parsed.append({"translations": {self.target_language: src}})

                except Exception as e:
                    self.logger.error(f"Batched translation request failed for items {i}-{i+len(batch)-1}: {e}")
                    for src in originals:
                        parsed.append({"translations": {self.target_language: src}})

        # At this point `parsed` should be a list of translation-like objects matching the input length
        try:
            self.logger.info(f"Parsed translations (count={len(parsed)}): {json.dumps(parsed if isinstance(parsed, list) else parsed, indent=2)[:2000]}")
        except Exception:
            pass

        # Update original data with translations where available
        for i, entry in enumerate(text_data.get("texts", [])):
            try:
                trans_obj = parsed[i]
                if not isinstance(trans_obj, dict):
                    continue
                translation = None
                # The per-entry results may be nested; try a couple of keys
                if "translations" in trans_obj and isinstance(trans_obj["translations"], dict):
                    # Try exact language key
                    translation = trans_obj["translations"].get(self.target_language)
                    # Try alias (ISO) if model returned short code like 'de'
                    if translation is None:
                        iso = self.lang_aliases.get(self.target_language)
                        if iso:
                            translation = trans_obj["translations"].get(iso)
                    # If still None, take any translation value present in the dict
                    if translation is None:
                        try:
                            # take first value in the translations mapping
                            translation = next(iter(trans_obj["translations"].values()))
                        except Exception:
                            translation = None
                if translation is None and self.target_language in trans_obj:
                    translation = trans_obj.get(self.target_language)
                if translation:
                    entry["translations"][self.target_language] = translation
                    self.logger.info(f"Added {self.target_language} translation for '{entry['original']}': '{translation}'")
                else:
                    self.logger.warning(f"No translation structure found for text: {entry['original']}")
            except IndexError:
                self.logger.error(f"No parsed translation available for index {i}; skipping")
            except Exception as e:
                self.logger.error(f"Error applying translation for '{entry.get('original', '')}': {e}")

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
            used_gpt_color = entry.get('font_info', {}).get('used_gpt_color', False)
            gpt_text_rgb = entry.get('font_info', {}).get('gpt_text_rgb')
            gpt_bg_rgb = entry.get('font_info', {}).get('gpt_bg_rgb')
            self.logger.info(f"Using stored text color {text_color} for text: {translated_text} (used_gpt_color={used_gpt_color}, gpt_text_rgb={gpt_text_rgb}, gpt_bg_rgb={gpt_bg_rgb})")
            
            # Load and scale font
            font_size = int(box_height * 0.8)
            font = None
            
            font_info = entry.get('font_info', {})
            custom_font_path = font_info.get('font_path')
            self.logger.info(f"Font info: {font_info}")
            
            if custom_font_path:
                if os.path.exists(custom_font_path):
                    try:
                        font = ImageFont.truetype(custom_font_path, font_size)
                        self.logger.info(f"Successfully loaded custom font: {custom_font_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to load custom font '{custom_font_path}': {str(e)}")
                        font = None
                else:
                    self.logger.error(f"Custom font path does not exist: {custom_font_path}")
            else:
                self.logger.warning("No custom font path provided in font_info")
                    
            if font is None:
                self.logger.info("Trying fallback fonts")
                fallback_tried = []
                for fallback in self.fallback_fonts:
                    try:
                        font = ImageFont.truetype(fallback, font_size)
                        self.logger.warning(f"Using fallback font: {fallback}")
                        break
                    except Exception as e:
                        fallback_tried.append(f"{fallback} ({str(e)})")
                        continue
                if font is None:
                    self.logger.error(f"All fallback fonts failed: {'; '.join(fallback_tried)}")
                        
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
    image_path = "test-images/test1.PNG"
    target_language = "german"
    
    translator = ImageTranslator(image_path, target_language)
    final_image_path = translator.process_image()
    print(f"Translation complete! Final image saved to: {final_image_path}")