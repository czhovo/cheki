import argparse
import base64
import io
import json
import math
import re
import sys
from datetime import date
from numbers import Real
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image, ImageOps


BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "date_polaroid_extraction_prompt.md"
API_KEYS_PATH = BASE_DIR / ".api_keys.json"
BASE_URL = "https://ws-9quka2z3ps3urphd.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.7-plus"
NORMALIZED_COORD_MAX = 1000
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
BOX_COLOR = (45, 170, 45)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Locate and transcribe one handwritten date with Qwen."
    )
    parser.add_argument("input", help="Input image path or directory.")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory for annotated PNG images.")
    parser.add_argument("--model", default=MODEL, help="Qwen model identifier.")
    parser.add_argument("--base-url", default=BASE_URL, help="Alibaba Model Studio compatible endpoint.")
    parser.add_argument("--api-keys-file", type=Path, default=API_KEYS_PATH, help="UTF-8 platform API-key JSON.")
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help="Thinking-token limit; use 0 to disable thinking.",
    )
    parser.add_argument("--max-tokens", type=int, default=1024, help="Final-answer token limit.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout in seconds.")
    parser.add_argument("--limit", type=int, help="Process only the first N sorted images.")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=PROMPT_PATH,
        help="UTF-8 system prompt file.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not request images whose annotated output already exists.",
    )
    return parser.parse_args()


def read_platform_key(path, platform):
    if not path.is_file():
        raise RuntimeError(f"API-key file does not exist: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    value = config.get(platform)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"API key is missing for platform: {platform}")
    return value.strip()


def image_sort_key(path):
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name.lower())


def iter_images(input_path):
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(
        (
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=image_sort_key,
    )


def load_image(path):
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def image_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def extract_date_json_object(content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model returned empty text.")
    text = content.strip()
    try:
        value = json.loads(text, strict=False)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model response does not contain a JSON object.")
        value = json.loads(text[start : end + 1], strict=False)
    if not isinstance(value, dict):
        raise ValueError("Model response JSON must be an object.")
    return value


def normalize_date_text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Date.text must be a non-empty string.")
    text = value.strip()
    full_date = re.fullmatch(r"(\d{4})\.(\d{2})\.(\d{2})", text)
    if full_date:
        year, month, day = (int(part) for part in full_date.groups())
        date(year, month, day)
        return text

    month_day = re.fullmatch(r"(\d{2})\.(\d{2})", text)
    if month_day:
        month, day = (int(part) for part in month_day.groups())
        date(2000, month, day)
        return text

    raise ValueError(
        f"Date.text must already use YYYY.MM.DD or MM.DD format: {value!r}"
    )


def normalize_result(value, image_size):
    if not isinstance(value, dict):
        raise ValueError("Result must be a JSON object.")
    extra_keys = set(value) - {"reasoning", "Date"}
    if "Date" not in value or extra_keys:
        raise ValueError(
            f"Result JSON must contain Date and may contain reasoning; extra keys: {sorted(extra_keys)}"
        )
    reasoning = value.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError("reasoning must be a string when present.")
    item = value["Date"]
    if item is None:
        return None
    if not isinstance(item, dict) or set(item) != {"bbox", "text"}:
        raise ValueError("Date must be null or an object containing only bbox and text.")

    box = item["bbox"]
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("Date.bbox must contain four coordinates.")
    if any(isinstance(part, bool) or not isinstance(part, Real) for part in box):
        raise ValueError("Date.bbox coordinates must be numeric.")
    x1, y1, x2, y2 = [float(part) for part in box]
    if not (
        0 <= x1 < x2 <= NORMALIZED_COORD_MAX
        and 0 <= y1 < y2 <= NORMALIZED_COORD_MAX
    ):
        raise ValueError(f"Date.bbox is outside normalized [0, 1000] coordinates: {box}")

    width, height = image_size
    pixel_box = [
        math.floor(x1 * width / 1000),
        math.floor(y1 * height / 1000),
        math.ceil(x2 * width / 1000),
        math.ceil(y2 * height / 1000),
    ]
    pixel_box[2] = min(width, pixel_box[2])
    pixel_box[3] = min(height, pixel_box[3])

    raw_text = item["text"]
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Date.text must be a non-empty string.")
    raw_text = raw_text.strip()
    try:
        normalized_text = normalize_date_text(raw_text)
        validation_error = None
    except ValueError as exc:
        normalized_text = None
        validation_error = str(exc)

    return {
        "bbox": pixel_box,
        "text": normalized_text or raw_text,
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "text_validation_error": validation_error,
    }


def request_date(client, model, prompt, data_url, image_size, thinking_budget, max_tokens):
    thinking_options = {"enable_thinking": thinking_budget > 0}
    if thinking_budget > 0:
        thinking_options["thinking_budget"] = thinking_budget

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                        "min_pixels": 65536,
                        "max_pixels": 2621440,
                    },
                ],
            },
        ],
        max_tokens=max_tokens,
        extra_body=thinking_options,
    )
    if not response.choices:
        raise ValueError("Model returned no choices.")
    return response.choices[0].message.content


def draw_date(image, result):
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if result is None:
        return bgr

    x1, y1, x2, y2 = result["bbox"]
    cv2.rectangle(bgr, (x1, y1), (x2 - 1, y2 - 1), BOX_COLOR, 5)

    label = result["text"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.05
    thickness = 3
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    label_x = max(0, min(bgr.shape[1] - text_width - 12, x1))
    if y1 >= text_height + baseline + 18:
        text_y = y1 - 10
    else:
        text_y = min(bgr.shape[0] - baseline - 5, y1 + text_height + 12)
    top = max(0, text_y - text_height - 8)
    bottom = min(bgr.shape[0] - 1, text_y + baseline + 5)
    cv2.rectangle(
        bgr,
        (label_x, top),
        (min(bgr.shape[1] - 1, label_x + text_width + 12), bottom),
        (255, 255, 255),
        -1,
    )
    cv2.putText(
        bgr,
        label,
        (label_x + 6, text_y),
        font,
        scale,
        BOX_COLOR,
        thickness,
        cv2.LINE_AA,
    )
    return bgr


def save_png(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode output image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    try:
        images = iter_images(input_path)
        if not images:
            raise RuntimeError(f"No supported images found in {input_path}")
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError("--limit must be positive.")
            images = images[: args.limit]
        prompt = args.prompt_file.resolve().read_text(encoding="utf-8")
        client = OpenAI(
            base_url=args.base_url,
            api_key=read_platform_key(args.api_keys_file.resolve(), "aliyun_bailian"),
            timeout=args.timeout,
            max_retries=0,
        )

        failures = []
        audit_path = output_dir / "responses.json"
        existing_audit = {}
        if args.skip_existing and audit_path.is_file():
            previous_records = json.loads(audit_path.read_text(encoding="utf-8"))
            if isinstance(previous_records, list):
                existing_audit = {
                    record["image"]: record
                    for record in previous_records
                    if isinstance(record, dict) and isinstance(record.get("image"), str)
                }
        audit_records = []
        for index, image_path in enumerate(images, start=1):
            output_path = output_dir / f"{image_path.stem}_annotated.png"
            if args.skip_existing and output_path.is_file():
                previous_record = existing_audit.get(image_path.name)
                if previous_record is not None and previous_record.get("raw_result") is not None:
                    audit_records.append(previous_record)
                else:
                    image = load_image(image_path)
                    raw_response_path = output_dir / "raw_responses" / f"{image_path.stem}.txt"
                    if raw_response_path.is_file():
                        raw_result = extract_date_json_object(
                            raw_response_path.read_text(encoding="utf-8")
                        )
                        result = normalize_result(raw_result, image.size)
                        audit_records.append(
                            {
                                "image": image_path.name,
                                "size": list(image.size),
                                "model": args.model,
                                "thinking_enabled": args.thinking_budget > 0,
                                "status": "recovered_raw_response",
                                "raw_result": raw_result,
                                "normalized_result": result,
                            }
                        )
                    else:
                        audit_records.append(
                            {
                                "image": image_path.name,
                                "size": list(image.size),
                                "model": args.model,
                                "thinking_enabled": args.thinking_budget > 0,
                                "status": "skipped_existing",
                                "raw_result": None,
                                "normalized_result": None,
                            }
                        )
                print(f"[{index}/{len(images)}] skipped existing {output_path}", flush=True)
                continue
            print(f"[{index}/{len(images)}] {image_path.name}", flush=True)
            try:
                image = load_image(image_path)
                raw_content = request_date(
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    data_url=image_data_url(image),
                    image_size=image.size,
                    thinking_budget=args.thinking_budget,
                    max_tokens=args.max_tokens,
                )
                raw_response_path = output_dir / "raw_responses" / f"{image_path.stem}.txt"
                raw_response_path.parent.mkdir(parents=True, exist_ok=True)
                raw_response_path.write_text(raw_content, encoding="utf-8")
                raw_result = extract_date_json_object(raw_content)
                result = normalize_result(raw_result, image.size)
                save_png(output_path, draw_date(image, result))
                audit_records.append(
                    {
                        "image": image_path.name,
                        "size": list(image.size),
                        "model": args.model,
                        "thinking_enabled": args.thinking_budget > 0,
                        "raw_result": raw_result,
                        "normalized_result": result,
                    }
                )
            except Exception as exc:
                failures.append((image_path, exc))
                print(f"[{index}/{len(images)}] failed: {exc}", file=sys.stderr, flush=True)
                continue
            state = "null" if result is None else result["text"]
            print(f"[{index}/{len(images)}] saved {output_path} ({state})", flush=True)

        output_dir.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(audit_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if failures:
            print(f"Failed: {len(failures)} of {len(images)} image(s).", file=sys.stderr)
            return 1
        print(f"Done: {len(images)} image(s) in {output_dir}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
