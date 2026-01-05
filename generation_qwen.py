import argparse
import torch
import pandas as pd
import os
import io
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
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
            except Exception as e:
                print(f"Failed to load image from path {first_img['path']}: {e}")

        if first_img.get("bytes") is not None:
            try:
                return Image.open(io.BytesIO(first_img["bytes"]))
            except Exception as e:
                print(f"Failed to load image from bytes: {e}")

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
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the input parquet file.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the inference results (parquet).")
    parser.add_argument("--lora_path", type=str,
                        help="Path to the local lora (used by vLLM).")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the local model (used by vLLM).")
    parser.add_argument("--cuda", type=str)
    parser.add_argument("--batch_size", type=int, default=30,
                        help="Batch size for inference.")
    parser.add_argument("--eval_mode", type=str, default='false',
                        help="Set Model to Eval")
    args = parser.parse_args()

    model_name = args.model_path

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=False)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.cuda}" if args.cuda else 'auto',
        trust_remote_code=False
    )
    if args.lora_path:
        model = PeftModel.from_pretrained(model, f"{args.lora_path}/actor/lora_adapter")
    if args.eval_mode == 'true':
        print(f'Model current status: Eval')
        model.eval()

    dataset = pd.read_parquet(args.data_path)[:]
    total = len(dataset)
    print(f"Total samples: {total}")

    all_responses = []
    batch_size = args.batch_size

    for start_idx in tqdm(range(0, total, batch_size)):
        end_idx = min(start_idx + batch_size, total)
        batch = dataset.iloc[start_idx:end_idx]

        texts = []
        images = []
        image_flags = []

        for row in batch.itertuples():
            chat = row.prompt.tolist() if hasattr(row.prompt, "tolist") else row.prompt

            image = load_image_from_row(row)

            if image is not None:
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": chat}
                    ]}
                ]
            else:
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": chat}]}
                ]

            text = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            texts.append(text)
            images.append(image)
            image_flags.append(image is not None)

        inputs = processor(
            text=texts,
            images=[img for img in images] if any(image_flags) else None,
            return_tensors="pt",
            padding=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                eos_token_id=processor.tokenizer.eos_token_id
            )

        gen_texts = processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        all_responses.extend(gen_texts)
        torch.cuda.empty_cache()

    dataset["responses"] = all_responses
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    dataset.to_parquet(args.output_path)
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main_task()
