from datasets import load_dataset
import os

# Create folders
os.makedirs("data/finance", exist_ok=True)
os.makedirs("data/marketing", exist_ok=True)
os.makedirs("data/general", exist_ok=True)

# --- FINANCE (concat 5 phrases per file) ---
finance_dataset = load_dataset("financial_phrasebank", "sentences_allagree", split="train")
for i in range(5):
    text = "\n\n".join(finance_dataset[j]['sentence'] for j in range(i * 5, (i + 1) * 5))
    with open(f"data/finance/doc_{i+1}.txt", "w") as f:
        f.write(text)

# --- MARKETING (filter + concat 5 items) ---
ag_dataset = load_dataset("ag_news", split="train")
marketing_texts = [item['text'] for item in ag_dataset if "marketing" in item['text'].lower()]

for i in range(5):
    chunk = marketing_texts[i*5:(i+1)*5]
    with open(f"data/marketing/doc_{i+1}.txt", "w") as f:
        f.write("\n\n".join(chunk))

# --- GENERAL (just chunk ag_news without filtering) ---
for i in range(5):
    chunk = [ag_dataset[j]['text'] for j in range(i*5, (i+1)*5)]
    with open(f"data/general/doc_{i+1}.txt", "w") as f:
        f.write("\n\n".join(chunk))

print("✅ Created 5 longer, paragraph-rich documents for each domain.")
