import json
import os
from datasets import Dataset
from PIL import Image
from tqdm import tqdm


def load_image(image_path):
    try:
        return Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {image_path}")
    except Exception as e:
        raise RuntimeError(f"加载图像时出错: {str(e)}")


def pre_process_dataset(split_name, path):
    ner_data = json.load(open(os.path.join(path, f"{split_name}.json"), 'r', encoding='utf-8'))
    for data in tqdm(ner_data):
        try:
            image = load_image(os.path.join(path, f"image/{data['id']}.jpg"))
        except Exception as e:
            image = None
        data['image'] = image
    return ner_data


def filter_factor(ner_data, factor=28):
    filtered = []
    for data in ner_data:
        if data['image'].size[0] >= factor and data['image'].size[1] >= factor:
            filtered.append(data)
    return filtered


def make_map_fn(split):
    def process_fn(example, idx):
        raw_text = example.pop("text")
        label = example.pop("label")
        images = [example.pop("image")]

        types = ['PER', 'LOC', 'ORG', 'MISC']
        prompt = instruction.format(types=', '.join(types), text=raw_text)

        gt = []
        for e in label:
            text = e['text']
            gt.append({
                "text": text,
                "type": e['type']
            })

        gt = {
            'ground_truth': gt,
            'text': raw_text,
            'seperator': '',
        }

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "images": images,
            "ability": "mner",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "split": split,
                "index": idx,
                "text": raw_text,
                "entities": label,
                "tokens": list(raw_text),
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    path_prefix = ''
    p = os.path.join(path_prefix, 'data/wmner')

    train_data = pre_process_dataset(split_name="train", path=p)
    val_data = pre_process_dataset(split_name="val", path=p)
    test_data = pre_process_dataset(split_name="test", path=p)

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)

    data_source = "wmner"
    instruction = open('./prompt.txt', 'r', encoding='utf-8').read()

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    train_dataset.to_parquet(os.path.join(path_prefix, 'data/wmner', "train.parquet"))
    val_dataset.to_parquet(os.path.join(path_prefix, 'data/wmner', "val.parquet"))
    test_dataset.to_parquet(os.path.join(path_prefix, 'data/wmner', "test.parquet"))
