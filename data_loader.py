import seaborn as sns
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "titanic.csv")

df = sns.load_dataset("titanic")
df.to_csv(OUTPUT_PATH, index=False)

print(f"Dataset save: {OUTPUT_PATH}")
print(f"Shape: {df.shape}")
