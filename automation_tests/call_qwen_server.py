#!/usr/bin/env python3
"""
Simple client to call the LUSTIFY image generation server
"""

import requests

SERVER_URL = "http://localhost:5055"


def generate_image(prompt, output_dir="single_call"):
    """Send a generation request to the server"""
    payload = {
        "prompt": prompt,
        "output_dir": output_dir
    }
    
    response = requests.post(f"{SERVER_URL}/t2i", json=payload)
    return response.json()


if __name__ == "__main__":
    import time

    #prompt = "photograph, real photo, young woman model, tan skin, pony tail blond hair, thin eyebrows, sexy eyes, long eyelash, sharp features, wearing a gold miniskirt, seductive provocative pose, cleavage, natural sunlight, 8k"
    #prompt = "photograph, real photo, young woman, tan skin, pony tail blond hair, thin eyebrows, sexy eyes, long eyelash, sharp features, gold miniskirt,provocative, facing viewer, natural sunlight, 8k"
    prefix = "candid photo, dim light, color graded portra 400 film, pores visible, remarkable detailed pupils, realistic dull skin noise, visible skin detail, dry skin"
    suffix = ", wide-shot, full-body view, bed"
    prompt = f"""{prefix},laugh, pubic hair, pale skin, high cheekbones,  platinum blond, dark blue eyes, long eyelashes, texture detail areola, {suffix}"""

    ts = int(time.time())
    output_path = f"{ts}_image.png"

    print(f"Sending request with prompt: {prompt}")
    result = generate_image(prompt, output_dir="single_call")
        
    if result:
            print(f"✅ Image generated: {result['image_path']}")
    else:
            print(f"❌ Failed: {result.get('error')}")


 