"""Local PP-OCRv5 mobile comparison on the Arabic pilot pages."""

import os

import pymupdf

from engines import normalize, result, word
from prepare import REPO, local_output


def extract(path, info, model_path, config):
    cache = local_output(REPO / config["working_dir"] / "setup/paddle_cache")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache))
    import numpy as np
    from paddleocr import PaddleOCR

    detection = "PP-OCRv5_mobile_det"
    recognition = "arabic_PP-OCRv5_mobile_rec"
    for name in (detection, recognition):
        if any(not (model_path / name / artifact).is_file()
               for artifact in ("inference.json", "inference.pdiparams", "inference.yml")):
            raise FileNotFoundError(f"Expected the offline PaddleOCR model {name}")
    model = PaddleOCR(
        text_detection_model_name=detection, text_detection_model_dir=str(model_path / detection),
        text_recognition_model_name=recognition, text_recognition_model_dir=str(model_path / recognition),
        device="cpu", cpu_threads=config["ocr_workers"], enable_mkldnn=False,
        use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
        return_word_box=False, text_det_limit_side_len=1600, text_det_limit_type="max",
    )
    with pymupdf.open(path) as pdf:
        pixmap = pdf[info["physical_page"] - 1].get_pixmap(dpi=config["dpi"], colorspace=pymupdf.csRGB)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
    [output] = model.predict(image[:, :, ::-1].copy())

    def box(polygon):
        points = np.asarray(polygon).reshape(-1, 2)
        return normalize([points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()],
                         pixmap.width, pixmap.height)

    texts = output["rec_texts"]
    lines = [[word(text, box(polygon))]
             for text, polygon in zip(texts, output["rec_polys"], strict=True)]
    scores = output["rec_scores"]
    return result(lines, text="\n".join(texts), diagnostics={
        "geometry_granularities": ["lines"],
        "ocr_mean_line_confidence": float(np.mean(scores)) if len(scores) else None,
        "ocr_lines": len(lines),
    })
