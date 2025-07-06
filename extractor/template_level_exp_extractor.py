import cv2
import numpy as np
import os
import asyncio
from .ai_extractor import preprocess_roi

# Configurations for template matching
TEMPLATES = {
    "level": {
        "path": os.path.join(os.path.dirname(__file__), '../templates/level-template.png'),
        "roi_offset_x": 5, "roi_width": 80, "roi_height": 40,
        "preprocess": "white_text"
    },
    "exp": {
        "path": os.path.join(os.path.dirname(__file__), '../templates/exp-template.png'),
        "roi_offset_x": 1, "roi_width": 150, "roi_height": 40,
        "preprocess": "adaptive"
    }
}

def template_match_extract(main_image, reader):
    """
    Extract 'level' and 'exp' using template matching from the given image.
    Returns: dict with keys 'level', 'exp' (if found)
    """
    main_image_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    results = {}
    for name, config in TEMPLATES.items():
        template = cv2.imread(os.path.abspath(config["path"]), cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue
        res = cv2.matchTemplate(main_image_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        match_threshold = 0.6
        if max_val < match_threshold:
            continue
        top_left = max_loc
        scale = 1  # No multi-scale
        tH, tW = template.shape[0], template.shape[1]
        scaled_offset_x = int(config["roi_offset_x"] * scale)
        scaled_width = int(config["roi_width"] * scale)
        scaled_height = int(config["roi_height"] * scale)
        roi_x = top_left[0] + tW + scaled_offset_x
        roi_y = top_left[1]
        roi = main_image[roi_y:roi_y + scaled_height, roi_x:roi_x + scaled_width]
        if roi.size == 0:
            continue
        preprocessed_roi = preprocess_roi(roi, config.get("preprocess", "simple"))
        allowlist = '0123456789[]%' if name == 'exp' else '0123456789,'
        ocr_result = reader.readtext(preprocessed_roi, detail=0, allowlist=allowlist)
        if ocr_result:
            full_text = "".join(str(x) for x in ocr_result)
            if name == "exp" and '[' in full_text:
                full_text = full_text.split('[')[0]
            cleaned_text = ''.join(filter(str.isdigit, full_text))
            results[name] = cleaned_text
    return results