import os

import huggingface_hub
from datasets import load_dataset

huggingface_hub.login()

SEA_11 = [
    "eng_Latn",
    "ind_Latn",
    "fil_Latn",
    "vie_Latn",
    "zsm_Latn",
    "khm_Khmr",
    "lao_Laoo",
    "mya_Mymr",
    "tha_Thai",
    "tam_Taml",
    "cmn_Hans",
]

out_dir = "/home/user/project/data/flores-plus_dev_devtest/"

os.makedirs(out_dir, exist_ok=True)

def write_split(ds, path, text_key="sentence"):
    with open(path, "w", encoding="utf-8") as f:
        for s in ds[text_key]:
            # Ensure single line per sentence
            s = s.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
            f.write(s + "\n")

for lang in SEA_11:
    # Load splits for this language
    # dev = load_dataset("openlanguagedata/flores_plus", name=lang, split="dev")
    devtest = load_dataset("openlanguagedata/flores_plus", name=lang, split="devtest")

    write_split(devtest, os.path.join(out_dir, f"{lang}.devtest"), text_key="text")
