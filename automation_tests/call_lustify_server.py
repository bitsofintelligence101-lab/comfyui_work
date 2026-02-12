#!/usr/bin/env python3
"""
Simple client to call the LUSTIFY image generation server
"""

import requests

SERVER_URL = "http://localhost:8052"


def generate_image(prompt, output_file="output.png", **kwargs):
    """Send a generation request to the server"""
    payload = {
        "prompt": prompt,
        "output_file": output_file,
        **kwargs
    }
    
    response = requests.post(f"{SERVER_URL}/generate", json=payload)
    return response.json()


if __name__ == "__main__":
    import time
    
    #Anime girl, black lace lingerie, green pleated skirt, thigh-high stockings, long black hair, green eyes, elegant indoor pose, soft lighting, fantasy style.
    #Anime girl, black lace bustier, green pleated skirt, thigh-high stockings, green eyes, long black hair, elegant indoor pose, natural lighting, soft focus, detailed textures.
    # Anime girl, black hair with clover, green eyes, wears green lace lingerie, black thigh-highs, grassy field, soft lighting, detailed skin, fantasy style, elegant pose 
    #Anime girl, black hair, green eyes, black lace top, green pleated skirt, thigh-highs, choker, grassy hills, sunset, soft lighting, detailed, vibrant colors, elegant pose
    
    #prompt = "Realistic anime, illustrious comic, grass meadow, laying on side, ((clevage)), real skin (irish) freckles, kiss, long straight black hair, white-eyes green iris, choker, (black lace lingerie) top, knees bent legs in air behind her, black knee high nylons, pleated green mini skirt, stilettos. sunlight rays, side view"
    
    # SDXL/Lustify prompt - proper Danbooru tag order:
    # rating -> style -> subjects -> character features -> clothing -> action -> setting
    #prompt = "rating:explicit, realistic, anime, 1boy, 1girl, hetero, kneeling, deepthroat, irrumatio, penis, oral, cum_in_mouth, freckles, long_hair, black_hair, green_eyes, breasts, nipples, choker, thighhighs, ejaculation, ahegao, saliva, grassy_field, outdoors, natural_lighting, pov"
    prompt = "photograph, 8k photo,1girl, long_hair, blue_eyes, brown_hair, straight_hair, no_mascara, no_eye_shadow, athletic, curvy, 1girl, bottomless, spread_legs, sex, hetero, missionary_position, pov, warm_golden_hour_lighting"
    ts = int(time.time())
    output_path = f"{ts}_image.png"

    print(f"Sending request with prompt: {prompt}")
    result = generate_image(prompt, output_file=output_path,shoot_folder="missionary_test")
        
    if result.get("success"):
            print(f"✅ Image generated: {result['output_path']}")
    else:
            print(f"❌ Failed: {result.get('error')}")


    location = "yacht"#hotel room, bedroom, cabin, castle, school, library, forest, 
    apperance_adj = "sexy"
    hair_color = "highlight"#blond, platinum, black, brown, highlight, red, pink, chestnut, auburn
    hair_style = f"two ponytail {hair_color} bangs"#f"short face framing {hair_color} hair", f"high ponytail {hair_color} hair", f"two ponytail {hair_color} hair" , f"wavy {hair_color} hair",  f"long {hair_color} hair", f"long {hair_color} hair"
    iris = "green"
    eye_color = f"subtle {iris} iris"
    eye_accents = "long-lashes, black mascara"# mascara, long-lashes, eye-shadow
    bone_structure = "sharp facial features"#soft features, round face, strong jaw, pointed chin, angular face
    clothing = "thighhighs" #nude, lingerie, bathrobe, miniskirt"
    body_type = "curvy, full breasts"#skinny, athletic, curvy, plump, hour-glass, fit, toned
    position = "lying on her back,vaginal sex, missionary_position, penetrate vagina, legs up"
    #"lying on her stomach, looking over shoulder,vaginal sex, sex_from_behind, he thrusts his penis in her vagina"
    face_prompt_enhancement = "highly detailed face,"
    angle_lighting = "from_above, soft lighting"

    prompts = [f"photograph, 8k photo,white,{face_prompt_enhancement}, {clothing}, {hair_style}, {eye_accents}, {eye_color},{bone_structure},{body_type},1girl 1boy, {position},{location}, {angle_lighting} "
,f"photograph, 8k photo, young, {face_prompt_enhancement}{clothing}, {hair_style}, {eye_accents}, {eye_color}, {bone_structure}, {body_type}, {position},1girl 1boy, {location},{angle_lighting}"
,f"photograph, 8k photo,teen, {face_prompt_enhancement}{clothing}, {hair_style}, {eye_accents}, {eye_color}, {bone_structure}, {body_type},1girl 1boy, {position}, {location},{angle_lighting}"
,f"photograph, 8k photo,milf, {face_prompt_enhancement}{clothing}, {hair_style}, {eye_accents}, {eye_color}, {bone_structure}, {body_type},1girl 1boy, {position},{location}, {angle_lighting}"
,f"photograph, 8k photo,old,{face_prompt_enhancement}{clothing}, {hair_style}, {eye_accents}, {eye_color}, {bone_structure}, {body_type},1girl 1boy, {position},{location}, {angle_lighting}"
,f"photograph, 8k photo,girl, {face_prompt_enhancement}{clothing}, {hair_style}, {eye_accents}, {eye_color}, {bone_structure}, {body_type},1girl 1boy, {position}, {location},{angle_lighting}"]

    
    
    
    for p in prompts:
        ts = int(time.time())
        output_path = f"{ts}_image.png"
        """prompt_file = f"{ts}_prompt.txt"
        with open(prompt_file, 'w') as f:
            f.write(prompt)"""
        
        print(f"Sending request with prompt: {p}")
        result = generate_image(p, output_file=output_path,shoot_folder="missionary_test")
        
        if result.get("success"):
            print(f"✅ Image generated: {result['output_path']}")
        else:
            print(f"❌ Failed: {result.get('error')}")
