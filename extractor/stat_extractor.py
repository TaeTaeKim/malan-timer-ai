import asyncio
import time
from .ai_extractor import extract_stats_from_image
from .template_level_exp_extractor import template_match_extract

def extract_all_stats_async(main_image, model, reader):
    """
    Run template matching (level, exp) and AI extractor (meso) in parallel.
    Returns: dict with keys 'level', 'exp', 'meso' (if found)
    """
    # loop = asyncio.get_event_loop()
    # Run template matching in thread pool (since it's CPU-bound)
    # template_result = template_match_extract(main_image, reader)
    # template_task = loop.run_in_executor(None, template_match_extract, main_image, reader)
    # Run AI extractor (already async)
    ai_result = extract_stats_from_image(main_image, model, reader)
    # ai_task = loop.run_in_executor(None, extract_stats_from_image,main_image, model, reader)
    # template_result, ai_result = await asyncio.gather(template_task, ai_task)
    return ai_result 