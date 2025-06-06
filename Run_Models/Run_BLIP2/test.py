import pandas as pd
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm
import os

# Setup
csv_file = "Cullen/labels.txt"  # The path must be manually set 
image_dir = "Cullen"  # The path must be manually set to appropriate folder

# Load data
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()

# Load BLIP-2
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
model.to(device)

# Store results for both prompts
meanings = []
gestures = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    image_path = os.path.join(image_dir, row["file name"])
    culture = row["culture"]

    try:
        image = Image.open(image_path).convert("RGB")

        # Prompt 1: What does the gesture mean?
        prompt_meaning = f"In the context of {culture} culture, what does this gesture mean?"
        inputs_meaning = processor(image, text=prompt_meaning, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)
        output_meaning = model.generate(**inputs_meaning, max_new_tokens=50)
        meaning = processor.decode(output_meaning[0], skip_special_tokens=True)

        # Prompt 2: What gesture is being made?
        prompt_gesture = f"In the context of {culture} culture, what gesture is the person making in this image?"
        inputs_gesture = processor(image, text=prompt_gesture, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)
        output_gesture = model.generate(**inputs_gesture, max_new_tokens=50)
        gesture = processor.decode(output_gesture[0], skip_special_tokens=True)

    except Exception as e:
        meaning = f"Error: {e}"
        gesture = f"Error: {e}"

    meanings.append(meaning)
    gestures.append(gesture)

# Save both outputs
df["blip2_meaning"] = meanings
df["blip2_gesture"] = gestures
df.to_csv("blip2_dual_prompt_output_I_M_P.csv", index=False)
print("✅ Dual-prompt gesture descriptions saved to blip2_dual_prompt_output.csv")

