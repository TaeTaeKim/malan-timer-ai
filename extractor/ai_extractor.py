import cv2
import re

def preprocess_roi(roi, method='simple'):
    import cv2
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if method == 'white_text':
        _, roi_binary = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY)
    elif method == 'adaptive':
        roi_binary = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        _, roi_binary = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY_INV)
    return cv2.resize(roi_binary, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)



def extract_stats_from_image(main_image, model, reader):
    """
    # This asynchronous function takes an input image, a YOLO model, and an OCR reader,
    # and extracts the "meso" stat from the image using object detection and OCR.
    # It returns a dictionary with the key "meso" and its extracted value (if found).
    """
    if main_image is None:
        raise ValueError("Invalid image data provided")
    extracted_data = {}
    results = model(main_image, conf=0.5, verbose=False)
    class_names = model.names
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            # if(class_name!="meso"): # 임시로 메소만 진행
            #     continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = main_image[y1:y2, x1:x2]
            # if class_name == "exp":
            #     h, w, _ = roi.shape
            #     crop_width = int(w * 0.23)
            #     roi = roi[:, crop_width:]
            if roi.size == 0:
                continue
            preprocess_method = 'white_text' if class_name == 'level' else 'simple'
            preprocessed_roi = preprocess_roi(roi, preprocess_method)
            allowlist = '0123456789,'
            if class_name == "exp":
                allowlist = '0123456789[]EXP%'
            ocr_raw_results = reader.readtext(preprocessed_roi, detail=0, allowlist=allowlist)
            if ocr_raw_results:
                full_text = "".join(str(x) for x in ocr_raw_results)
                if class_name == "exp":
                    if '[' in full_text:
                        full_text = full_text.split('[')[0]
                ocr_raw_results = re.sub(r'[^0-9]', '', full_text)
            if ocr_raw_results:
                extracted_data[class_name] = ocr_raw_results
    return extracted_data

