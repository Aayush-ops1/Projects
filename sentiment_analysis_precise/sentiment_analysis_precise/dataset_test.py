from datasets import load_dataset

dataset = load_dataset("dair-ai/emotion")

print(dataset)

print("\nTrain sample:\n")
print(dataset["train"][0])

print("\nLabels:\n")
print(dataset["train"].features["label"].names)