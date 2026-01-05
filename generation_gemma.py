import argparse
import torch
import pandas as pd
import os
import io
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from tqdm import tqdm
from peft import PeftModel


def load_image_from_row(row):
    if not hasattr(row, "images") or not row.images:
        return None

    first_img = row.images[0]

    if isinstance(first_img, dict):
        if first_img.get("path"):
            try:
                with Image.open(first_img["path"]) as img:
                    return img.copy()
            except Exception:
                return None

        if first_img.get("bytes") is not None:
            try:
                return Image.open(io.BytesIO(first_img["bytes"]))
            except Exception:
                return None

    if isinstance(first_img, str):
        try:
            with Image.open(first_img) as img:
                return img.copy()
        except Exception:
            return None

    return None


def main_task():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cuda", type=str)
    parser.add_argument("--eval_mode", type=str, default="false")
    args = parser.parse_args()

    # === 1. 加载 Gemma 3 Vision ===
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=False
    )

    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.cuda}" if args.cuda else "auto",
        trust_remote_code=False
    )

    if args.lora_path:
        if os.path.exists(f"{args.lora_path}/actor/lora_adapter"):
            model = PeftModel.from_pretrained(model, f"{args.lora_path}/actor/lora_adapter")
        else:
            model = PeftModel.from_pretrained(model, f"{args.lora_path}")

    if args.eval_mode == "true":
        model.eval()

    dataset = list(pd.read_parquet(args.data_path).T.to_dict().values())
    total = len(dataset)
    print(f"Total samples: {total}")

    all_responses = []

    for item in tqdm(dataset):
        chat = item['prompt'][0]['content']
        image = f"path"

        messages = [
            {"role": "user", "content": [
                {"type": "text", 'text': chat},
                {"type": "image", 'image': image}
            ]},
        ]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            generation = model.generate(
                **inputs,
                max_new_tokens=1024
            )
            generation = generation[0][input_len:]

        gen_texts = processor.decode(generation, skip_special_tokens=True)
        all_responses.append(gen_texts)
        torch.cuda.empty_cache()

    df = pd.read_parquet(args.data_path)
    df["responses"] = all_responses
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_parquet(args.output_path)
    print(f"Done! Results saved to {args.output_path}")


if __name__ == "__main__":
    main_task()
