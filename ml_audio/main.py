from datasets import load_dataset

minds = load_dataset(
    "PolyAI/minds14", name="en-AU", split="train", trust_remote_code=True
)

print(minds)
