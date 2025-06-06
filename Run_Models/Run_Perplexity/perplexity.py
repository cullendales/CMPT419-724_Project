import os
import requests
import json

#################### Script 1 ####################
folder_path = "Indian_Mexican_Portuguese"
caption = "In the context of Indian culture, what gesture is this person making and what it means. Explain in one single sentence."
outputs = {}

for i in range(1, 16):
    image_name = f"image{i}.png"
    image_path = os.path.join(folder_path, image_name)    
    with open(image_path, "rb") as image_file:
        response = requests.post(
            "https://api.perplexity.ai/v1/sonnet",  
            files={"image": image_file},
            data={"caption": caption},
            headers={"Authorization": "API_KEY"}  
        )
        if response.status_code == 200:
            outputs[image_name] = response.json()       
    

with open("results.json", "w") as f:
    json.dump(outputs, f, indent=2)

print("Complete")

#################### Script 2 ####################

folder_path = "Indian_Mexican_Portuguese"
caption = "In the context of Mexican culture, what gesture is this person making and what it means. Explain in one single sentence."
outputs = {}

for i in range(16, 31):
    image_name = f"image{i}.png"
    image_path = os.path.join(folder_path, image_name)    
    with open(image_path, "rb") as image_file:
        response = requests.post(
            "https://api.perplexity.ai/v1/sonnet",  
            files={"image": image_file},
            data={"caption": caption},
            headers={"Authorization": "API_KEY"}  
        )
        if response.status_code == 200:
            outputs[image_name] = response.json()       
    

with open("results.json", "w") as f:
    json.dump(outputs, f, indent=2)

print("Complete")

#################### Script 3 ####################

folder_path = "Indian_Mexican_Portuguese"
caption = "In the context of Portuguese culture, what gesture is this person making and what it means. Explain in one single sentence."
outputs = {}

for i in range(31, 41):
    image_name = f"image{i}.png"
    image_path = os.path.join(folder_path, image_name)    
    with open(image_path, "rb") as image_file:
        response = requests.post(
            "https://api.perplexity.ai/v1/sonnet",  
            files={"image": image_file},
            data={"caption": caption},
            headers={"Authorization": "API_KEY"}  
        )
        if response.status_code == 200:
            outputs[image_name] = response.json()       
    

with open("results.json", "w") as f:
    json.dump(outputs, f, indent=2)

print("Complete")

#################### Script 4 ####################

folder_path = "Canadian_Korean_Vietnamese"
caption = "In the context of Canadian culture, what gesture is this person making and what it means. Explain in one single sentence."
outputs = {}

for i in range(1, 17):
    image_name = f"c-gesture{i}.png"
    image_path = os.path.join(folder_path, image_name)    
    with open(image_path, "rb") as image_file:
        response = requests.post(
            "https://api.perplexity.ai/v1/sonnet",  
            files={"image": image_file},
            data={"caption": caption},
            headers={"Authorization": "API_KEY"}  
        )
        if response.status_code == 200:
            outputs[image_name] = response.json()       
    

with open("results.json", "w") as f:
    json.dump(outputs, f, indent=2)

print("Complete")

#################### Script 5 ####################

folder_path = "Canadian_Korean_Vietnamese"
caption = "In the context of Korean culture, what gesture is this person making and what it means. Explain in one single sentence."
outputs = {}

for i in range(1, 17):
    image_name = f"k-gesture{i}.png"
    image_path = os.path.join(folder_path, image_name)    
    with open(image_path, "rb") as image_file:
        response = requests.post(
            "https://api.perplexity.ai/v1/sonnet",  
            files={"image": image_file},
            data={"caption": caption},
            headers={"Authorization": "API_KEY"}  
        )
        if response.status_code == 200:
            outputs[image_name] = response.json()       
    

with open("results.json", "w") as f:
    json.dump(outputs, f, indent=2)

print("Complete")

#################### Script 6 ####################

folder_path = "Canadian_Korean_Vietnamese"
caption = "In the context of Vietnamese culture, what gesture is this person making and what it means. Explain in one single sentence."
outputs = {}

for i in range(1, 17):
    image_name = f"v-gesture{i}.png"
    image_path = os.path.join(folder_path, image_name)    
    with open(image_path, "rb") as image_file:
        response = requests.post(
            "https://api.perplexity.ai/v1/sonnet",  
            files={"image": image_file},
            data={"caption": caption},
            headers={"Authorization": "API_KEY"}  
        )
        if response.status_code == 200:
            outputs[image_name] = response.json()       
    

with open("results.json", "w") as f:
    json.dump(outputs, f, indent=2)

print("Complete")

#################### Script 7 ####################

folder_path = "Persian_Italian_French"
caption = "In the context of Persian culture, what gesture is this person making and what it means. Explain in one single sentence."
outputs = {}

for i in range(1, 11):
    image_name = f"image{i}.png"
    image_path = os.path.join(folder_path, image_name)    
    with open(image_path, "rb") as image_file:
        response = requests.post(
            "https://api.perplexity.ai/v1/sonnet",  
            files={"image": image_file},
            data={"caption": caption},
            headers={"Authorization": "API_KEY"}  
        )
        if response.status_code == 200:
            outputs[image_name] = response.json()       
    

with open("results.json", "w") as f:
    json.dump(outputs, f, indent=2)

print("Complete")

#################### Script 8 ####################

folder_path = "Persian_Italian_French"
caption = "In the context of Italian culture, what gesture is this person making and what it means. Explain in one single sentence."
outputs = {}

for i in range(11, 28):
    image_name = f"image{i}.png"
    image_path = os.path.join(folder_path, image_name)    
    with open(image_path, "rb") as image_file:
        response = requests.post(
            "https://api.perplexity.ai/v1/sonnet",  
            files={"image": image_file},
            data={"caption": caption},
            headers={"Authorization": "API_KEY"}  
        )
        if response.status_code == 200:
            outputs[image_name] = response.json()       
    

with open("results.json", "w") as f:
    json.dump(outputs, f, indent=2)

print("Complete")

#################### Script 9 ####################

folder_path = "Persian_Italian_French"
caption = "In the context of French culture, what gesture is this person making and what it means. Explain in one single sentence."
outputs = {}

for i in range(28, 47):
    image_name = f"image{i}.png"
    image_path = os.path.join(folder_path, image_name)    
    with open(image_path, "rb") as image_file:
        response = requests.post(
            "https://api.perplexity.ai/v1/sonnet",  
            files={"image": image_file},
            data={"caption": caption},
            headers={"Authorization": "API_KEY"}  
        )
        if response.status_code == 200:
            outputs[image_name] = response.json()       
    

with open("results.json", "w") as f:
    json.dump(outputs, f, indent=2)

print("Complete")