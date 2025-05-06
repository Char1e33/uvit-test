import torch
import os
import sys

sys.path.append("/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/jxy_22307130086/uvit-base")
import numpy as np

import sys

print(sys.path)

import libs.autoencoder
import libs.clip
from datasets import MSCOCODatabase
import argparse
from tqdm import tqdm


def main():
    prompts = []
    VIS_NUM = 32
    target_dir = "scripts/Vis_caps/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True,
                        help="指定要使用的Bert模型版本, 例如:'StanfordAIMI/RadBERT'")
    args = parser.parse_args()

    version = args.version

    dirlist = os.listdir("/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/public/T2I_datasets/ISIC/ISIC2018/response_910_test")
    for i, file in enumerate(dirlist):
        if i >= VIS_NUM:
            break
        file_path = os.path.join("/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/public/T2I_datasets/ISIC/ISIC2018/response_910_test", file)

        with open(file_path, 'r') as f:
            content = f.read()
            prompts.append(content)

        output_path = os.path.join(target_dir, f"{i}.txt")
        with open(output_path, 'w') as out_f:
            out_f.write(content)

    device = 'cuda'
    clip = libs.clip.BertEmbedder(version)
    clip.eval()
    clip.to(device)

    save_dir = f'assets/datasets/ISIC2018-256_features-{version.split("/")[-1]}/run_vis'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    latent = clip.encode(prompts)
    for i in range(len(latent)):
        c = latent[i].detach().cpu().numpy()
        np.save(os.path.join(save_dir, f'{i}.npy'), {"prompt": prompts[i], "context": c}, allow_pickle=True)
    print("finished: extract test prompt features")


if __name__ == '__main__':
    main()