import pandas as pd
import os
from tqdm import tqdm
import argparse

from gpt.t2i import call_LLM as gpt_call
from nano_banana.t2i import call_LLM as nano_call
from doubao.t2i import call_LLM as doubao_call


def get_func_and_save_dir(model_type, save_root):
    if model_type == 'gpt':
        func = gpt_call
    elif model_type == 'nano_banana':
        func = nano_call
    elif model_type == 'doubao':
        func = doubao_call

    save_dir = os.path.join(save_root, f"{model_type}_variation_2")
    os.makedirs(save_dir, exist_ok=True)

    return func, save_dir


def test_var_1(
    model_type,
    save_root = "/home/hong/hongyu/violin/test_results",
    var_json_path = "/home/hong/hongyu/violin/benchmark/metadata/Variation_2_metadata.csv",
):

    df = pd.read_csv(var_json_path, usecols=['id', 'prompt'])

    func, save_dir = get_func_and_save_dir(model_type, save_root)

    for index, row in tqdm(df.iterrows()):
        current_id = row['id']
        current_prompt = row['prompt']

        save_path = os.path.join(save_dir, f"id_{current_id}.png")

        if os.path.exists(save_path):
            continue

        func(current_prompt, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Variation-2 variations.")
    parser.add_argument(
        '--model', 
        type=str, 
    )
    args = parser.parse_args()

    test_var_1(model_type=args.model)