#after testing on 2019 intel-5 MBP: the model fails while loading checkpoints due to memory (Llava is very large)
#works on Nvidia GPU
#used this websites as reference: https://colab.research.google.com/drive/1qsl6cd2c8gGtEW1xV5io7S8NHh-Cp1TV?usp=sharing#scrollTo=W48r3NxDRskb

from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# using llava but bakllava could potentially work as it requires less resources
model_id = "llava-hf/llava-1.5-7b-hf"
#model_id = "llava-hf/bakLlava-v1-hf"

model = LlavaForConditionalGeneration.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)

#testing first on a single image to test if the model loads properly - rename any of the images to gesture.jpg to test
test_image = Image.open("gesture.jpg")

#prompt varies based on culture
prompt = "In the context of Canadian culture, what gesture is the person making in this image"
inputs = processor(images=test_image, text=prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        do_sample=False
    )

result = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)

