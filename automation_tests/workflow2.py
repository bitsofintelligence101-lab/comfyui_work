"""
FIRST: run qwen_server.py to start the local API server.

Then run this workflow2.py script to execute the workflow.

Orchestration workflow for full prompt to video.
1) Create initial character model traits from text prompt.
2) Create scene concept description from text prompt.
3) Generate action, camera angles, and shots from scene concept from text prompt for the character
4) Generate Images for each shot using the character model traits and scene concept.
5) review images, reject and regenerate as needed.
6) Compile images into video.
7) compare start image, end image, prompt to determine if final video is acceptable.
8) add audio
9) combine all the above into final video file.
"""

#Workflow steps use local API endpoints to perform each step.
# e.g. http://localhost:5055/generate
import requests
import time
import re
from PIL import Image
import os
import random
from comfy_call import ComfyCall
import json
from join_mp4 import combine_videos
from deployed import gstorage
from video_optical_flow import calculate_optical_flow_with_grid
import concurrent.futures
import torch

URL_BASE = "http://localhost:5055"
GCLOUD_STORAGE_BUCKET = "x"
COMFY = ComfyCall()#connect to comfy server

# PRIMARY WORKFLOW FUNCTIONS

def generate_audio_clips(video_clips, prompts, start_images, folder, base_file_name):
    """
    Generate audio for a list of video clips.
    
    Args:
        video_clips: List of video file paths
        prompts: List of animation prompts corresponding to each video
        start_images: List of start images used for each video clip
        folder: Target folder for uploads
        base_file_name: Base filename for outputs
    
    Returns:
        List of video paths with audio (same length as input video_clips)
    """
    global GCLOUD_STORAGE_BUCKET
    video_clips_with_audio = []
    bucket_name = GCLOUD_STORAGE_BUCKET

    _ = _unload_model()#unload chat model to free vram
    _ = _unload_image_generate()#unload image gen model to free vram, should already be unloaded
    _ = _unload_sfx_model() #unload soundfx model to free vram, should already be unloaded

    
    for idx, vid_path in enumerate(video_clips):
        # Skip audio for scene transitions
        if idx < len(prompts) and "scene change transition" in prompts[idx].lower():
            print(f"Skipping audio for clip {idx+1} (scene transition)")
            video_clips_with_audio.append(vid_path)
            continue
        
        try:
            print(f"Generating SoundFX for clip {idx+1}/{len(video_clips)}: {vid_path}...")
            
            # Get corresponding prompt and image
            prompt_str = prompts[idx] if idx < len(prompts) else ""
            img_path = start_images[idx] if idx < len(start_images) else ""
            
            # Generate audio prompt
            sfx_p = soundfx_prompt(img_path, prompt_str)
            print(f"Audio Prompt: {sfx_p}")

            # Save audio prompt to file
            try:
                output_dir = os.path.dirname(vid_path)
                audio_prompt_filename = f"{base_file_name}_{idx + 1}_audio_prompt.txt"
                audio_prompt_path = os.path.join(output_dir, audio_prompt_filename)
                
                with open(audio_prompt_path, "w", encoding="utf-8") as f:
                    f.write(sfx_p)
                    
                # Upload prompt to GCS
                if os.path.exists(audio_prompt_path):
                    gstorage.upload_blob(bucket_name, audio_prompt_path, f"{folder}/{audio_prompt_filename}")
            except Exception as e:
                print(f"Error saving audio prompt: {e}")
            
            # Generate audio video
            final_v = soundfx_generate(sfx_p, vid_path)
            print(f"Generated audio video path: {final_v}")
            
            
            if final_v and os.path.exists(final_v):
                # Upload audio video to GCS
                audio_video_filename = os.path.basename(final_v)
                gstorage.upload_blob(bucket_name, final_v, f"{folder}/{audio_video_filename}")
                video_clips_with_audio.append(final_v)
            else:
                video_clips_with_audio.append(vid_path)
                
        except Exception as e:
            print(f"Error generating audio for clip {idx+1}: {e}")
            video_clips_with_audio.append(vid_path)
    
    return video_clips_with_audio

def handle_scene_transition(source_image_path, transition_prompt, folder, source_image_review, source_image_prompt, 
                           counter, lora, clear_vram, base_file_name, height, width, transition_frames=33, 
                           max_transition_attempts=3, return_transition_video=False):
    """
    Handle scene change transitions by generating a new starting image.
    Purpose: Create a transition frame that will be used as the starting image for the next animation.
    Note: The video is not saved to the final output, only the final image is used as the next start image.
    
    Args:
        source_image_path: Path to the original source image
        transition_prompt: The prompt containing 'scene change transition'
        folder: Target folder for outputs
        source_image_review: Review of the source image
        source_image_prompt: Prompt used to generate source image
        counter: Current clip counter
        lora: LORA tag to use
        clear_vram: Whether to clear VRAM
        base_file_name: Base filename for outputs
        height: Image height
        width: Image width
        transition_frames: Number of frames for transition (default 33)
        max_transition_attempts: Maximum retry attempts (default 3)
        return_transition_video: If True, return tuple (image_path, video_path); if False, return only image_path (default False)
    
    Returns:
        If return_transition_video=False: Path to the transition image if successful, None if all attempts failed
        If return_transition_video=True: Tuple of (transition_image_path, transition_video_path) if successful, (None, None) if all attempts failed
    """
    transition_review_passed = False
    wan_transition_result = None

    #remove SLOW motion or Fast Motion from the start of the prompt to avoid confusion
    transition_prompt = re.sub(r'^(slow motion|fast motion)[\.,;:!\- ]*', '', transition_prompt.lower(), flags=re.IGNORECASE).strip()
    
    for transition_attempt in range(max_transition_attempts):
        print(f"\n\nCreating scene transition frame (attempt {transition_attempt + 1}/{max_transition_attempts})...PROMPT: ", transition_prompt)
        wan_transition_result = i2v(
            image_path=source_image_path, 
            prompt=transition_prompt, 
            folder=folder,
            source_image_review=source_image_review, 
            source_image_prompt=source_image_prompt,
            file_name_modifier=f"{counter + 1}_transition" if transition_attempt == 0 else f"{counter + 1}_transition_retry{transition_attempt}", 
            lora=lora, 
            clear_vram=clear_vram,
            base_file_name=base_file_name, 
            height=height, 
            width=width,
            length=transition_frames
        )
        
        if not wan_transition_result or 'image_final' not in wan_transition_result:
            print(f"ERROR: Transition generation failed on attempt {transition_attempt + 1}")
            continue
        
        transition_image = wan_transition_result['image_final']
        print("Transition image generated: ", transition_image)
        
        # Create custom review prompt for scene transition
        transition_review_prompt = f"""You are reviewing a scene transition image. This image was generated from the prompt: {transition_prompt}

                The question is does this image look like what the end result of the prompt would be?

                CRITICAL REQUIREMENTS FOR SCENE TRANSITION [Quality]:
                1. POSE: Does the woman's pose match the intended scene described in the prompt? Check if she is in the correct position (standing/kneeling/laying/sitting) as described.
                2. FACE: Her face MUST be clearly visible and facing the camera. Eyes MUST be open with direct eye contact.
                3. CLOTHING: Check if her clothing state matches the prompt. If the prompt indicates 'naked' or 'nude' or specific body parts visible (breasts, nipples, vagina, etc.), verify those are visible. If clothing is mentioned, verify it's present. 
                4. ANATOMY: Anatomically correct. No distortions, correct number of fingers if hands visible, no extra limbs.

                This is a transition frame that will be used as the starting image for the next animation clip, so accuracy is critical. nudity or clothed, face clearly visible, all critical.

                Return [Quality] if the image meets ALL criteria and explain why.
                If it fails ANY criteria, return [Non-Quality] and explain exactly what is wrong."""
        
        # Review the transition image
        transition_review = image_review(transition_image, transition_prompt, review_prompt=transition_review_prompt)
        
        if '[quality]' in transition_review.lower():
            print(f"\nTransition image review PASSED on attempt {transition_attempt + 1}")
            print(f"Review feedback: {transition_review}")
            transition_review_passed = True
            break
        else:
            print(f"\nTransition image review FAILED on attempt {transition_attempt + 1}")
            print(f"Review feedback: {transition_review}")
            
            if transition_attempt < max_transition_attempts - 1:
                print("Retrying transition generation...")
    
    # Return the transition image (and optionally video) if we have one
    if wan_transition_result and 'image_final' in wan_transition_result:
        if not transition_review_passed:
            print("WARNING: Using transition image that did not pass review after all attempts")
        
        if return_transition_video:
            transition_video = wan_transition_result.get('video', None)
            return wan_transition_result['image_final'], transition_video
        else:
            return wan_transition_result['image_final']
    else:
        print(f"ERROR: Failed to generate transition image after all {max_transition_attempts} attempts")
        if return_transition_video:
            return None, None
        else:
            return None

def videogen_all_prompts_provided(image_path="", prompts=[],folder="neon",source_image_review="", source_image_prompt="", lora="sex",base_file_name=None, physical_description=""):
    global COMFY, URL_BASE, GCLOUD_STORAGE_BUCKET
    #generate animations for each of the prompts provided.
    #ANIMATION Dimentions
    height = 720#1024# 640 #800 480#
    width = 1280#1024# 640 #800 720#
    transition_frames = 33 #number of frames to use for scene transitions
    source_image_path = image_path
    total_prompts = len(prompts)
    print(f"Total prompts to generate: {total_prompts}")
    clear_vram = False #default when starting video gen to clear vram between clips
    counter = 0
    start_image = image_path
    all_files = []#list of all result dicts
    video_clips = []#list of all video clip paths
    if base_file_name is None:
        base_file_name = str(int(time.time()))
    previous_image = image_path
    start_images = []  # Track start images for audio generation

    # Generate all video clips first
    while counter < total_prompts:
        #TODO:
            #
            #Add logic to detect faces and their positions to maintain continuity (pretrained YOLO model? use VLM?)
            #
            #
            # Use VLM to check if the prompt makes sense for the image
            #
            #  
            print(f"Generating clip {counter+1} of {total_prompts}...")
            start_time = time.time()
                
            wan_result = None
            review_passed = False
            #must parse the LORA from the prompt if present
            lora, clean_prompt = extract_lora_tag(prompts[counter])
            prompts[counter] = clean_prompt
            current_prompt = prompts[counter]

        
            if counter == 0:
                # First clip - generate without review (no previous animation to evaluate)
                print("USING ANIMATION PROMPT: ", current_prompt)
                if lora is None:
                    lora = 'sex'
                
                wan_result = i2v(image_path=start_image, prompt=current_prompt, folder=folder,source_image_review=source_image_review, source_image_prompt=source_image_prompt,file_name_modifier=f"{counter + 1}", lora=lora, clear_vram=clear_vram,base_file_name=base_file_name, height=height, width=width)
                review_passed = True
            else:
                # Generate animation first, then review the result
                max_review_attempts = 3
                
                print("\n\nOriginal Animation Prompt: ", current_prompt)
                last_result = None  # Keep track of last generation result

                #IF scene transition, add the physical description to the prompt to help continuity
                if "scene change transition" in current_prompt.lower():
                    transition_prompt = re.sub(r'^(slow motion|fast motion)[\.,;:!\- ]*', '', current_prompt.lower(), flags=re.IGNORECASE).strip()
                    if lora.lower() == 'transition':
                        #TODO: Replace with qwen-image-edit designed for this purpose of reposing or undressing the character
                        # Use the extracted function to handle scene transition
                        transition_image = handle_scene_transition(
                            source_image_path=source_image_path,
                            transition_prompt=transition_prompt,
                            folder=folder,
                            source_image_review=source_image_review,
                            source_image_prompt=source_image_prompt,
                            counter=counter,
                            lora=lora,
                            clear_vram=clear_vram,
                            base_file_name=base_file_name,
                            height=height,
                            width=width,
                            transition_frames=transition_frames
                        )
                        
                        if transition_image:
                            print("Using image ", transition_image, "as starting image for clip ", counter+1)
                            print("Scene transition complete. Moving to next prompt without generating animation for transition prompt.")
                            start_image = transition_image
                            counter += 1
                            continue  # Skip to next prompt - transition prompt doesn't generate a video clip
                        else:
                            print(f"ERROR: Failed to generate acceptable transition image. Continuing with current start_image.")
                
                if not review_passed:
                    #add the facial and physical description to the prompt to help continuity if her eyes were closed, or face was obscured, etc.
                    current_prompt = f"""{current_prompt}. Maintain these details of her features: {physical_description}"""
                    print("\n\nFeature Modified Animation Prompt: ", current_prompt)
                    
                    # Store the original intended prompt before optical flow analysis
                    original_intended_prompt = current_prompt
                    
                    #Run Optical Flow analysis and prompt review on the start_image to guide animation prompt and adjust if needed
                    # Use -1 to get the last video clip since transitions don't add videos to the list
                    motion_helped_prompt = create_informed_animation_prompt(video_path=video_clips[-1], intended_prompt=current_prompt, scene_number=counter+1, all_scene_prompts=prompts, grid_size=5, output_file=f"{base_file_name}_optical_flow_{counter}.png")
                    
                    # If motion analysis suggests a scene change transition, try to recover by generating transition and retrying
                    max_transition_recovery_attempts = 3
                    transition_recovery_attempt = 0
                    
                    #Unintended scene change call from the AI, this was not planned in the original prompt sequence, AI thinks we need to try to recover by generating a transition frame to achieve the desired animation
                    while "scene change transition" in motion_helped_prompt.lower() and transition_recovery_attempt < max_transition_recovery_attempts:
                        print(f"\n!!! Motion analysis suggested scene change transition was needed (recovery attempt {transition_recovery_attempt + 1}/{max_transition_recovery_attempts})")
                        print(f"Transition prompt suggested: {motion_helped_prompt}")
                        print("Attempting to generate transition image and re-analyze with ORIGINAL intended prompt...")
                        
                        # Extract LORA from the transition prompt
                        transition_lora, transition_clean_prompt = extract_lora_tag(motion_helped_prompt)
                        
                        # Generate the transition image AND video for optical flow analysis
                        transition_image, transition_video = handle_scene_transition(
                            source_image_path=source_image_path,
                            transition_prompt=transition_clean_prompt,
                            folder=folder,
                            source_image_review=source_image_review,
                            source_image_prompt=source_image_prompt,
                            counter=counter,
                            lora=transition_lora if transition_lora else 'transition',
                            clear_vram=clear_vram,
                            base_file_name=base_file_name,
                            height=height,
                            width=width,
                            transition_frames=transition_frames,
                            return_transition_video=True
                        )
                        
                        if transition_image:
                            print(f"Transition image created: {transition_image}")
                            print(f"Transition video created: {transition_video}")
                            print(f"Updating start_image and re-analyzing with ORIGINAL intended prompt: {original_intended_prompt}")
                            start_image = transition_image
                            #remove 'scene change transition' from the motion helped prompt before re-analysis else it will automatically suggest it again
                            motion_helped_prompt = motion_helped_prompt.lower().replace("scene change transition", "")
                            
                            # Re-run optical flow analysis with the transition video and ORIGINAL intended prompt
                            # Use the transition video for optical flow analysis to understand motion from transition
                            #to better understand what is happening here, the AI determined that to acheive the desired prompt, based on the input image from the previous clip, we had to do a scene change transition that was not expected.
                            #we have a new image now, that should better match the intended prompt, so we re-analyze using the original intended prompt with the transition image to see if it worked and then updating the prompt with motion
                            #queues again since our starting image has changed.
                            motion_helped_prompt = create_informed_animation_prompt(
                                video_path=transition_video if transition_video else video_clips[-1], 
                                intended_prompt=original_intended_prompt, 
                                scene_number=counter+1, 
                                all_scene_prompts=prompts, 
                                grid_size=5, 
                                output_file=f"{base_file_name}_optical_flow_{counter}_recovery{transition_recovery_attempt}.png"
                            )
                            print(f"\nRe-analysis result: {motion_helped_prompt}\n")
                            
                            # Check if re-analysis is now successful (no longer suggests transition)
                            if "scene change transition" not in motion_helped_prompt.lower():
                                print(f"SUCCESS: Re-analysis no longer suggests scene change after attempt {transition_recovery_attempt + 1}. THis means we got a good starting image for the prompt. Proceeding with animation.")
                                print("Using image ", transition_image, "as starting image for clip ", counter+1)
                                print("Scene transition complete. Moving to next prompt without generating animation for transition prompt.")
                                start_image = transition_image
                                break # Exit recovery loop and continue with animation generation using our updated start image

                        if transition_recovery_attempt == max_transition_recovery_attempts:
                            print("ERROR: Failed to generate transition image during recovery. Breaking recovery loop and using last generated image")
                            start_image = transition_image if transition_image else start_image
                            break
                        
                        transition_recovery_attempt += 1
                    
                    
                    print(f"\n\nImage Analysis Adjusted Animation Prompt: {motion_helped_prompt}\n\n")
                    print("extract lora tag from prompt")
                    lora, current_prompt = extract_lora_tag(motion_helped_prompt)#returns the lora tag, and the prompt with the tag removed
                    print("Using LORA: ", lora)
                    
                    for attempt in range(max_review_attempts):
                        print(f"\n--- CLIP {counter+1} of {total_prompts}... Generation Attempt {attempt + 1} of {max_review_attempts} ---")
                        print("\nANIMATION PROMPT: ", current_prompt)
                        
                        # Generate the animation first
                        wan_result = i2v(image_path=start_image, prompt=current_prompt, folder=folder, source_image_review=source_image_review, source_image_prompt=source_image_prompt, file_name_modifier=f"{counter + 1}" if attempt == 0 else f"{counter + 1}_retry{attempt}", lora=lora, clear_vram=clear_vram, base_file_name=base_file_name, height=height, width=width)
                        
                        # Store whatever we got, if retry exceeds max attempts just keep last
                        if wan_result:
                            last_result = wan_result
                        
                        # Check if generation succeeded
                        if not wan_result or 'image_final' not in wan_result:
                            print(f"ERROR: Generation failed on attempt {attempt + 1}")
                            continue
                        
                        # Now review the result (end frame) against the prompt we used
                        end_frame = wan_result['image_final']
                        print("\nImage Review Disabled for testing purposes. Assuming quality result.\n")
                        #img_review = image_review(end_frame, current_prompt)
                        img_review = '[quality] - image review was disabled, all first attempts assumed quality'  # For testing, assume quality
                        source_image_review = img_review
                        
                        if '[quality]' in img_review.lower():
                            print("\nReview passed! Result is quality.\n")
                            print("Review feedback: ", img_review)
                            review_passed = True
                            break
                        else:
                            print(f"Review failed. Result did not meet quality standards.")
                            print(f"Review feedback: {img_review}")
                            
                            #attempt is 0 indexed, so -1 for max attempts
                            if attempt < max_review_attempts - 1:
                                # Adjust prompt based on review feedback for next attempt
                                adjustment_prompt = f"""This Prompt was: {current_prompt}.\n Was used to generate an animation that transitioned from Image1 to Image2 provided.\n
                                THERE WAS AN ERROR, the automatic system reviewed the final image (Image2) and determined it did not meet the quality standards for animation based on the prompt provided.
                                You must adjust the prompt to improve the animation quality based on the review feedback provided, so we can achieve the desired animation.
                                    CRITICAL RULES FOR ANIMATION SCENES:
                                    - The scene prompt must be simple, focused on the motion.
                                    - Maximum 25 words per scene.
                                    - Focus ONLY on continuous motion.
                                    - Describe physical movements in detail: body parts moving, expressions changing, physical interactions
                                    - Use verbs like: swaying, thrusting, bouncing, arching, trembling, gliding, panning, zooming.
                                    - Use his with penis so the model with render the man with it
                                    - Keep her face in frame
                                However, this review of Image2 indicates it was not a quality result: {img_review}\n 
                                Adjust the prompt so we achieve the desired quality animation of Image1. CRITICAL IMPORTANT: if the animation was transitioning between actions (from blowjob to sex, from striptease to blowjob, from sex to ejaculation, etc.) You should put: 
                                \n'Scene Change Transition. Image goes dark. Then we show the same woman...' at the start of the prompt. This is the only way to make abrupt changes consistently. RETURN a revised prompt ONLY."""
                                revised_prompt = qwen_generate(adjustment_prompt, images=[start_image, end_frame])
                                
                                print("\nRevised Prompt for next attempt: ", revised_prompt)
                                current_prompt = revised_prompt
                    
                    # Use whatever result we have (last attempt or last successful one)
                    if last_result is not None:
                        wan_result = last_result
                    
                    if not review_passed:
                        print(f"WARNING: Max review attempts reached for clip {counter + 1}. Using last generated result.")
            
            #Final animation result dict, has all the file paths
            result = wan_result

            # Check if result is empty or missing required keys
            if not result:
                print(f"ERROR: i2v returned empty result for clip {counter}")
                raise ValueError(f"Video generation failed for clip {counter} - empty result returned")
            if 'video' not in result:
                print(f"ERROR: i2v result missing 'video' key for clip {counter}. Keys present: {list(result.keys())}")
                raise ValueError(f"Video generation failed for clip {counter} - no video in result")
            

            
            all_files.append(result)
            video_clips.append(result['video'])
            start_images.append(start_image)  # Track start image for audio generation
            print(video_clips)

            # Upload video and animation prompt to GCS
            bucket_name = GCLOUD_STORAGE_BUCKET
            if 'video' in result and os.path.exists(result['video']):
                video_path = result['video']
                video_filename = os.path.basename(video_path)
                gstorage.upload_blob(bucket_name, video_path, f"{folder}/{video_filename}")
                
            if 'animation_prompt' in result and os.path.exists(result['animation_prompt']):
                anim_prompt_path = result['animation_prompt']
                anim_prompt_filename = os.path.basename(anim_prompt_path)
                gstorage.upload_blob(bucket_name, anim_prompt_path, f"{folder}/{anim_prompt_filename}")

            counter += 1

            #get the final image path
            end_image = result['image_final']
            previous_image = start_image
            start_image = end_image

            clear_vram = False #only clear vram on first clip or if model change

            end_time = time.time()
            print(f"\n\nClip {counter} generation completed in {end_time - start_time:.2f} seconds.\n\n")

    # Extract base filename from first video (all share same unique number prefix)
    base_file_name = os.path.splitext(os.path.basename(video_clips[0]))[0].rsplit('_', 1)[0]
    output_dir = os.path.dirname(video_clips[0])
    
    # 1. Combine and Upload Silent Videos
    print(f"Combining all silent video clips into a single video...{video_clips}")
    full_video_filename_silent = os.path.join(output_dir, f"{base_file_name}_full_video.mp4")
    combine_videos(video_clips, full_video_filename_silent)
    
    # Upload full video to GCS
    bucket_name = GCLOUD_STORAGE_BUCKET
    if os.path.exists(full_video_filename_silent):
        full_video_basename_silent = os.path.basename(full_video_filename_silent)
        gstorage.upload_blob(bucket_name, full_video_filename_silent, f"{folder}/{full_video_basename_silent}")

    generate_audio = False
    if generate_audio:
        # 2. Generate Audio for All Clips
        print("\n=== Starting Audio Generation for All Clips ===")
        video_clips_with_audio = generate_audio_clips(video_clips, prompts, start_images, folder, base_file_name)
        
        # 3. Combine and Upload Audio Videos
        print(f"Combining all audio-enhanced video clips into a single video...{video_clips_with_audio}")
        full_video_filename_audio = os.path.join(output_dir, f"{base_file_name}_full_video_sfx.mp4")
        combine_videos(video_clips_with_audio, full_video_filename_audio)
        if os.path.exists(full_video_filename_audio):
            full_video_basename_audio = os.path.basename(full_video_filename_audio)
            gstorage.upload_blob(bucket_name, full_video_filename_audio, f"{folder}/{full_video_basename_audio}")
    else:
        print("\n\nAudio generation disabled, skipping audio generation step.\n")
        
    return all_files    


def workflow_all_together(image_steering = None,shoot_folder="all_together_test5", image_gen_sys_prompt = None, animation_sys_prompt = None, animation_prompt = None, prefix = None, suffix=None, source_image_prompt=""):
    #Initial Prompt - Generate an image prompt
    #image_steering is either, none which means we will generate a character description from scratch
    #or a text description of the character to be generated
    #or a path to an image file to be used as reference for character generation

    _ = _load_chat_model()
    
    if image_gen_sys_prompt is None or animation_sys_prompt is None or animation_prompt is None:
        print("ERROR: System prompts for image generation and animation generation must be provided. Also need an animation prompt.")
        raise ValueError("System prompts for image generation and animation generation must be provided.")
        
    ts = None
    img_path = None
    character_description = None
    
    #STEP 1: Create or Use Character Description to generate initial image, or use provided image
    if image_steering is None or type(image_steering) == str and not os.path.isfile(image_steering):
        img_gen_attempts = 3
        img_prompt = ""
        while img_gen_attempts > 0:
            #set the system prompt
            print("\nSETTING SYSTEM PROMPT for IMAGE GENERATION\n")
            _ = set_system_prompt(image_gen_sys_prompt)

            #prompt to generate character description, from scratch or with traits
            if prefix is None:
                prefix = "8k, soft lighting, candid cinema, 16mm, color graded portra 400 film, skin pores visible skin detail, remarkable detailed pupils, realistic dull skin noise"
            if suffix is None:
                suffix = "wide-shot, her full-body"

            if image_steering is None:
                img_prompt = f"""Create a detailed description of a woman character, for an erotic photoshoot, in a location of your choice.
                The description part of the prompt can not exceed 15 words and the total prompt length can not exceed 35 words (prefix and suffix).
                it MUST start with prefix: '{prefix}'.  and end with suffix: '{suffix}'.  Here are some examples that you can use as inspiration:
                \n"{prefix}, young woman, blond ponytail hair, green eyes, long eyelashes, tall, sagging breasts, white lingerie and panties. a luxury hotel room, {suffix}"
                \n"{prefix}, 18 yo, short face-framing pink hair, long eyelashes, strong facial features, leather skirt, boots, printed t-shirt, smile, a desert cabana with a bed, {suffix}"
                \n"{prefix}, girl, wavy brown hair, green eyes, long lashes, dimpled smile, curvy, natural breasts, red lace lingerie and thigh highs, tropical bungalow curtains, {suffix}"
                \n"{prefix}, 20 yo, auburn hair in a ponytail, blue eyes, freckles, athletic build, small breasts, sports bra and shorts, urban rooftop at sunset, {suffix}"
                \n"{prefix}, young woman, light freckled skin, flowing black hair, blue eyes, long eyelashes, tall, silk blouse pencil skirt, corner office with view, {suffix}"
                \n"{prefix}, young woman, short platinum blond hair, glasses, blue-grey eyes, yellow bikini, luxury yacht deck, {suffix}"
                Remember total prompt length can not exceed 35 words.
                Other features like glasses, tattoos, piercings, hair color, breast size, skin tone, clothing style are all fair game. sexy clothing is encouraged. miniskirts, lingerie, tight dress, yoga pants, leggings, sequin dress, etc.
                Return <image_prompt></image_prompt> tags with the full image prompt, and <physical_description></physical_description> tags with ONLY her physical description (no clothing, location, or pose). """
            else:
                img_prompt = f"""Create a detailed description of a woman character that has THESE TRAITS {image_steering}, for an erotic photoshoot, in a location of your choice.
                The description part of the prompt can not exceed 15 words and the total prompt length can not exceed 35 words (prefix and suffix).
                it MUST start with prefix: '{prefix}'.  and end with suffix: '{suffix}'.   Here are some examples that you can use as inspiration:
                \n"{prefix}, young woman, blond ponytail hair, green eyes, long eyelashes, tall, sagging breasts, white lingerie and panties. a luxury hotel room, {suffix}"
                \n"{prefix}, 18 yo, short face-framing pink hair, long eyelashes, strong facial features, leather skirt, boots, printed t-shirt, smile, a desert cabana with a bed, {suffix}"
                \n"{prefix}, girl, wavy brown hair, green eyes, long lashes, dimpled smile, curvy, natural breasts, red lace lingerie and thigh highs, tropical bungalow curtains, {suffix}"
                \n"{prefix}, 20 yo, auburn hair in a ponytail, blue eyes, freckles, athletic build, small breasts, sports bra and shorts, urban rooftop at sunset, {suffix}"
                \n"{prefix}, young woman, light freckled skin, flowing black hair, blue eyes, long eyelashes, tall, silk blouse pencil skirt, corner office with view, {suffix}"
                Remember total prompt length can not exceed 35 words.
                sexy clothing is encouraged. miniskirts, lingerie, tight dress, yoga pants, leggings, sequin dress, etc. Use the suggested traits to influence the character design.
                Return <image_prompt></image_prompt> tags with the full image prompt with incorporated TRAITS, and <physical_description></physical_description> tags with ONLY her physical description (no clothing, location, or pose). """
            
            img_prompt_tagged = qwen_generate(img_prompt, model_type='vlm')
            source_image_prompt = extract_image_tag(img_prompt_tagged)
            character_description,_ = extract_physical_description_tag(img_prompt_tagged)
            print(f"Generated Character Description Prompt:\n{img_prompt_tagged}\n\n")
            #now create the image, get the path
            img_path = image_generate(source_image_prompt, directory=shoot_folder)

            if not img_path:
                print("Failed to generate image. Retrying...")
                img_gen_attempts -= 1
                continue

            #review the image
            review_prompt = f"""You review the base image used in a photoshoot. The image must show a single female character.
            CRITICAL REQUIREMENTS FOR [Quality]:
            1. FACE: Her face MUST be highly detailed, clearly visible, and facing the camera. No distortion in eyes or features. This is critical for character consistency in animations.
            2. POSE: She must be facing the camera directly or very slightly angled. The shot must be a Half-Body (waist up) or Full-Body portrait. No extreme close-ups, no back shots. We need to see her body to re-pose her later.
            3. ANATOMY: Anatomically correct. Hands must have 5 fingers. No extra limbs.
            4. VISIBILITY: At minimum, her upper body is fully visible. If she is wearing clothing, it must be clear what she is wearing.
            
            This is a foundational image for a video workflow. If the face is obscured, blurry, or not facing the camera, it fails. If the body is not visible enough to determine her outfit/physique, it fails.
            
            Return [Quality] if it passes all criteria, as well as why.
            If it does not pass return [Non-Quality] and explain exactly which criteria failed."""
            review_result = image_review(img_path, source_image_prompt, review_prompt=review_prompt)

            if "[quality]" in review_result.lower() :
                print("Initial image passed quality review.")
                break
            else:
                print("Initial image did not pass quality review, regenerating...")
                img_gen_attempts -= 1
            
            if img_gen_attempts == 0:
                print("Warning: Initial image did not pass quality review after 3 attempts, proceeding anyway with last generated image.")

        # Check if we have a valid image path before proceeding
        if img_path is None:
            print("Error: Failed to generate initial image after all attempts. Cannot proceed.")
            return None
        
        #take the time stamp from the first part of the file name, image_generate adds timestamp prefix automatically
        ts = int(os.path.basename(img_path).split("_")[0])
        _unload_image_generate()
        print("unloaded image generation model to free VRAM\n\n")
                

    else:
        #img_path is the provided image
        try:
            if source_image_prompt == "":
                _ = set_system_prompt("You create detailed descriptions of characters based on images provided. ONLY describe her physical appearance, do NOT describe clothing, background, or pose.\n ONLY discuss what you do say. return a natural language description of her physical traits.\n \
                like: 'long flowing blond hair, blue eyes, curvy body, large breasts, pale skin, freckles, long eyelashes, angular facial features, full lips, gold hoop earings'.")
                print("No source image prompt provided, will attempt to create one from the image. THIS DOESN'T WORK WELL.\n")
                #STEP 2: Create a description of the character
                #TODO: THIS DOESN"T WORK!!! It is not detecting the image features well.  Go back to pulling this from the initial prompt or have user provide it.
                #print("\n\nFIX NEEDED! Generating Character Physical Description from image...doesn't work well\n")
                character_description_prompt = f"""You are an expert character designer. Your goal is to create a precise physical description of the woman in this image so she can be recreated in other scenes.
                Analyze the image and provide a short comma-separated list of her physical traits inside <physical_description></physical_description> tag.
                
                FOCUS ON THESE SPECIFIC PHYSICAL TRAITS:
                1. FACE: Detailed description of facial features, shape, eyes (color & shape), lips, nose, makeup style. accessories like glasses, earrings, necklace.
                2. HAIR: Exact color, specific style (e.g. messy bun, straight long, curly bob), and texture.
                3. BODY: Body type (e.g. slim, curvy, athletic), bust size, hips, and general physique.
                4. SKIN: Skin tone, texture (e.g. freckles, tan, pale, white, milano, dark), and any marks.

                DO NOT INCLUDE:
                CLOTHING: Specific style, color, and type of clothing she is wearing. We only care about her physical appearance.
                BACKGROUND/POSE: Ignore any details about the background or her pose in the image.
                
                CRITICAL: 
                - Keep the description descriptive but concise enough to be used as a stable diffusion prompt. 
                - Do NOT describe the background or pose or clothing, ONLY her physical appearance.
                - IGNORE lighting and camera angles.
                
                Return ONLY face, hair, body type, skin type description inside tag: <physical_description>...</physical_description>"""
                character_description = qwen_generate(character_description_prompt, images=[img_path], model_type='vlm')
                character_description, _ = extract_physical_description_tag(character_description)
                print(f"Extracted Character Description from provided image:\n{character_description}\n\n")
                print("RESETTING SYSTEM PROMPT after Character Description Extraction\n")
                _ = set_system_prompt("")#reset system prompt

            #take the time stamp from the first part of the file name, if possible
            ts = int(os.path.basename(img_path).split("_")[0])
        except Exception as e:
            print(f"Error extracting timestamp from image path: {e}, adding one and resaving")
            ts = int(time.time())
            #load image and resave with timestamp prefix
            img = Image.open(image_steering)
            new_image_path = os.path.join(shoot_folder, f"{ts}_image.png")
            img.save(new_image_path)
            img_path = new_image_path

        img_path = image_steering
        
    print(f"Creating seires under ts: {ts} with image: {img_path}\n\n")
    
    #set the system prompt
    print("\nSETTING SYSTEM PROMPT for ANIMATION GENERATION\n")
    _ = set_system_prompt(animation_sys_prompt)

    #STEP 3: Create Scenes
    if animation_prompt is None:
        print("No animation prompt provided (the one that generates scenes), the default one is NO GOOD. please provide one.\n")
        animation_prompt = f"""You are an expert erotic photographer. You are creating a detailed scene description for an erotic photoshoot."""
    
    attempts_ = 3
    animation_prompt_list = []

    while attempts_ > 0:
        #call AI - retry up to 3 times to ensure scene transitions are present
        required_phrase2 = "<lora>ejaculation</lora>"
        required_phrase = 'scene change transition'
        animation_result = None
        for retry in range(3):
            #provide the image to help with context
            animation_result = qwen_generate(animation_prompt, images=[img_path])

            if required_phrase in animation_result.lower() and required_phrase2 in animation_result.lower():
                print(f"Required scene transition phrase found on attempt {retry + 1}")
                break
            else:
                print(f"Attempt {retry + 1}/3: Required phrase not found, retrying...")
        else:
            print("Warning: Required scene transition phrase not found after 3 attempts, proceeding anyway with prompt results")
        
        #now send for animation, get all the animations
        animation_prompt_list = extract_scenes(animation_result)
        print(f"Extracted Animation {len(animation_prompt_list)} Prompts: \n\n{animation_prompt_list}\n\n")
        
        if isinstance(animation_prompt_list,list):
            break
        attempts_ -= 1

    #reset system prompt to null
    _ = set_system_prompt("")
    print("\nRESETTING SYSTEM PROMPT to NULL, Starting Animation Generation Loop\n")

    #STEP 4: Generate Animations for each scene prompt
    #should now have an image and list of prompts to send to video generation flow
    allFiles = videogen_all_prompts_provided(image_path=img_path, prompts=animation_prompt_list, folder=shoot_folder, source_image_review="NONE", source_image_prompt=source_image_prompt, lora="sex",base_file_name=str(ts), physical_description =character_description)
    print("Done")
    return allFiles


#########
#
# Auxiliary Convienience Functions
#
#########
def soundfx_prompt(img_path,animation_prompt):
    #takes an image path and comes up with a prompt for the sound fx for the scene.
    sfx_prompt = """You come up with a short sound effect description which are used as a prompt for a video to sound AI model. You are creating sound for the image, which was animated with this animation prompt: """ + animation_prompt + """\n\n
    1. Add key sounds effects that would enhance the erotic atmosphere.
    2. Use descriptive language to convey the intensity and mood of the scene.
    3. Keep it concise.
    4. Include specific sexual sound elements like 'moans, breathing, orgasim, ejaculation, sucking, vaginal intercourse, thrusting, slapping, gagging, deepthroat, kissing, laughing, giggling, whispering, blowjob, intercourse, skin contact, etc.'
    5. Avoid generic terms like "sexy sounds" or "erotic noises".
    6. When adding environmental ambient background sounds add them at the end of the prompt.
    7. Keep the description focused on sounds only, short and to the point.
    8. You can create some ambient background sounds if it fits the scene, like nature sounds, city noises, ocean waves, etc.
    9. Ensure the prompt is appropriate for generating sound effects only. DO NOT add music or songs.
    10. Graphic sexual sounds are encouraged to match the explicit nature of the scene.

    Use clear and descriptive keywords to specify the desired sounds. For example, instead of just "Water," use "Gentle waves lapping against shore."

    ONLY return language text prompt, no special tags, characters, formatting, asterisks, or quotes.

    First review the image and determine what sexual activity is happening, the intensity level, and the characters involved.
    Then create the sound effect prompt based on the image and animation prompt provided.

    Return only the sound effect prompt that matches the image and animation prompt."""
    result = qwen_generate(sfx_prompt, images=[img_path])
    return result

def image_review(img_path,image_prompt,review_prompt=None):

    if review_prompt is None:
        review_prompt = f"""you review sexually explicit images with mature content. The image you are reviewing was the LAST frame image from a video clip generated from this prompt:{image_prompt}.\
        Critically inspect and determine if there were any errors in anatomy.\
        The most frequent errors anatomical errors.\
        \nInspect the image carefully it may be hard to determine some of these questions.\
        FOR IMAGES WITH PENIS:
        \nIf you see a penis, is it attached to a mans body,or torso or thighs or a full male body? The man does not have to be visible in the image, but the penis should look like it attached to a body even if the body is only slightly visible.\
        \n Has the penis disapeared, this is obvious when a torso, body, pubic area between legs is visible but the penis is missing. THIS IS ONLY for images where the woman character is in the same image as a man or partial body of a man.

        GENERAL ANATOMICAL CHECKS:
        \nIf you see hands do they have the proper number of fingers?\
        \nIs every arm, leg attached to a body in the correct orientation? No limbs are not bending backwards, neck and head are not rotated unnaturally.\
        \nThere are no extra limbs (legs, feet, arms, etc.)\
        \nThere should only be one of 3 options: 1 woman, or 1 woman and 1 man, or a woman with a man partially visible.\
        \nPhysical poses are correct, nothing anatomically impossible?\
        \nIf you see a visible vagina is clearly defined with detail and labia.\
        \nIf there are any abnormalities or it has features that fail any of these criteria, image has NOT met standards and is not a quality image.\
        
        PERSPECTIVE CHECK:
        \nIf sex is happening. Make sure the position makes sense. Either a POV view or a logical angle that matches the described scene. Penis can't enter a thigh, side of worman, stomach, back, etc. it can only enter her mouth, her vagina or her anus.

        CHARACTER CHECK:
        \nOnly one women is in the scene? There CAN NOT be more than one woman.
        Finally, This is VERY important. If all of the other criteria have been met, Is the face of the woman visible? She must be looking at viewer (camera). Her face MUST be visible to be [Quality]\
        \n
        IS THE WOMAN's FACE VISIBLE? IT MUST BE to be quality. It is critical for this. DO NOT pass the image if you can't see her face and eyes
        \n
        Determine if this image is quality or not.\
        Return you assesment and end with [Quality] or [Not-Quality] in your assessment"""

    result = qwen_generate(review_prompt, images=[img_path])
    return result

def create_informed_animation_prompt(video_path, intended_prompt="",scene_number=0,all_scene_prompts=[], grid_size=5, frame_offset =4, output_file="optical_flow_analysis.png",analysis_prompt=None, physical_attributes=None):
    #Calculate optical flow with grid
    #by default calculate optical flow will downsize image to 640 x640
    optical_flow_img, optical_flow_path = calculate_optical_flow_with_grid(video_path, grid_size=grid_size, frame_offset=frame_offset, output_path=output_file)
    #Run Optical Flow analysis between previous_image and start_image to guide prompt adjustments
    print(f"Optical flow image saved to: {optical_flow_path}")
    #now that we have the optical flow image, use the VLM to analyze the image and create an updated prompt.
    if analysis_prompt is None:
        analysis_prompt = f"""You are an expert animation prompt specialist. 
        You are working on a single scene clip of a larger video. 
        The animations are done using an Image-To-Video (I2V) AI model that is highly sensitive to the input image.
        It uses Last Frame image as the source image to the next frame.  You are reviewing a modified version of the last frame image that has motion vectors overlaid on it to help you understand how the character was moving leading up to this frame.
        Your job is to help adjust the intended animation prompt to adapt to any motion or positions of the subjects shown in the image.

        GENERAL PROMPT STRUCTURE GUIDE:
            Motion speed - overall speed of action FAST or SLOW
            Cast and count - main character details, number of characters
            Setting and time - location, environment, time of day
            Clothing and appearance - outfit details, physical traits usually starts with 'Maintain these details of her features: ...'
            Camera and framing - angle, shot type (zoom-in/out, dolly-in/out, orbit, pan, tilt, upper body, full body, etc)
            Action  (what they do) - detailed actions and movements MOST IMPORTANT with the I2V model
            LORA tag - to help the I2V model understand the type of motion to apply

        You are reviewing a modified version of the last frame input image, motion blur and green arrow motion vectors have been added to help you understand what was happening.\n

        HOW TO INTERPRET THIS  IMAGE:
            1. The reference image shows motion vectors as green arrows, the larger the arrow the faster the motion in that direction.\n
            2. Focus on the green arrows in the image to determine motion speed and hints about direction.\n
            3. Focus on green arrows on the main character's body parts (arms, legs, head, torso) to determine motion., not background elements.\n
            4. Use the SPECIFIC direction words in the prompt. For example, 'head looking up', 'left leg kicking right', etc.\n
            5. Integrate motion hints for body parts with significant motion (large arrows). To animate more naturally.\n
            6. Use logical motion hints that make sense with the image content and intended animation prompt. The motion blur with the green arrows should help you determine this.\n
            7. YOU MUST remember ends of limbs (hands, feet) motion appear much faster than body motion. e.g. the arrows must be very large on ends of limbs to be considered significant fast motion.\n
            8. ALWAYS state action speed (FAST many large arrows/SLOW many small arrows) overall then update the intended animation prompt as needed for more seemless motion, or transitions like where hands, head, legs, feet, etc. are going to.
            9. If it appears Clothing is being removed make sure the resulting naked body parts are described as being visible (shirt/top/bra removed, breasts and nipples visible. skirt/dress/panties removed, vagina and pubic area visible) and the clothing is discarded\n
            10. Use 'forward' and 'backward' or 'downward' or 'upward' for limb movements when hinting them. forward and backward and rotation tend to be better animation indicators that 'up' or 'down' for arms.
            11. DO NOT leave awkward or strange poses.  Sometimes the womans will be straight up, make sure to prompt them to go back down or to do something with them but DO NOT leave them extended in to the air.
        
        If the intended animation prompt with this image is too different from what the image shows, 
        the Image-To-Video (I2V) model will produce an animation it 'thinks' fits the image rather than follow the prompt, or it will have generally poor results.  
        This means we MAY need to adjust the prompt to better fit the image.

        INPUT IMAGE IMPORTANT CONTEXT:
        The image you are reviewing shows motion vectors as green arrows overlaid on a motion blurred image. The I2V model doesn't see the blur or green arrow vectors 
        The temporal blur is just to help visualize motion, and the green arrows are the critical part for the direction of motion. 
        The temporal motion blur was created by overlaying the final frames of the previous animation with reduced opacity for the older in time frames.
        This helps visualize how the character was moving leading up to this image. And therefore helps you determine if the intended animation prompt fits the image or not.

        BE AWARE that you are working in an itterative process, so it is OK to output a prompt that is similar, or revert to a scene reset transition prompt if needed.  
        This will allow the next pass of your attempt to animiate to the desired Intended Animation Prompt to be more accurate.
        We want to acheive the overall flow of the scenes as listed below but it is OK if we need to adjust the intended animation prompt to better fit.

        The provided image and prompt are for scene number {scene_number} out of {len(all_scene_prompts)} scenes in the overall video. The Intended Animation Prompt for the provided image is: '{intended_prompt}'. The full flow of the scenes and all of the past and future prompts are as follows:
        {all_scene_prompts}

        Again, The Intended Animation Prompt for the provided image is: '{intended_prompt}'.

        The woman's physical attributes are: {physical_attributes}. Make sure you can see these attributes in the image or you may need to use a 'scene change transition' prompt to reset her appearance.

        Notice that all scene prompts Must MUST have a LORA tag at the end of the prompt, inside of the scene tag, to help the I2V animation model properly animate. tag choices are:
        -- <lora>standard</lora> for general movements and striptease scenes
        -- <lora>masturbation</lora> for female masturbation scenes
        -- <lora>sex</lora> for any missionary type intercourse sexual act scenes
        -- <lora>doggystyle</lora> for doggy style intercourse sexual act scenes viewed from the side
        -- <lora>blowjob</lora> for blowjob scenes   
        -- <lora>ejaculation</lora> for any or ejaculation scenes (facial, on body, on breasts, etc) this is used at the end typically
        -- <lora>transition</lora> for when the scene transitions to a new pose in anticipation of a new scene, used with the 'Scene Change Transition...' prompt
        \n\n
        THE ANALYSIS PARAMETERS:
        FIRST - determine if the Intended Animation Prompt is a good fit for the provided image. If it is, return it unchanged.
        HOW TO DETERMINE IF THE INTENDED PROMPT IS A GOOD FIT:
        1. Is the main character's pose in the image consistent with the actions described in the Intended Animation Prompt?
        Examples:
        -- If the prompt is for animating missionary sex is she laying down on her back looking at viewer in a POV style angle? 
        -- If the prompt is for animating a standing pose is she standing up straight facing the viewer?
        -- If the prompt is for animating a kneeling pose or kneeling blowjob is she kneeling on a surface facing the viewer? 
        -- If the prompt is for animating doggy style sex, is she viewed from the side on her hands and knees?
        -- If the prompt indicates she should be naked, does she have clothing on still?
        -- Is her face visible, eyes open making eye contact with viewer? This is a CRITICAL review. If the face is not, then we CAN NOT attempt to show her face in the animation because the I2V model does not know what she looks like and will make up a new face.

        IF THE INTENDED PROMPT IS A GOOD FIT:
        Return the Intended Animation Prompt unchanged. Or slight changes to indicate how different body parts are moving based on the optical flow image.
        For example, if the optical flow image shows her head moving upward quickly, you can add 'head looking upward quickly' to the prompt.
        Or for example if the optical flow image shows her left arm moving forward quickly, and she is supposed to be adjusting her clothing, you can add 'left arm reaching forward quickly to adjust her clothing' to the prompt.
        or if she is supposed to be nude and you see her wearing clothes, you can add 'she removes her clothes, then specificy nudity result [breasts and nipples visible, vagina visible, etc.] to the prompt.

        IF THE INTENDED PROMPT IS NOT A GOOD FIT:
        You have two options:
        OPTION 1 IF THE INTENDED PROMPT IS NOT A GOOD FIT- Modify the Intended Animation Prompt slightly to better fit the image.
        Make small adjustments to the prompt to better align with the character's pose and visibility in the image.
        For example, if the prompt indicates she is standing but she is clearly kneeling in the image, change 'standing' to 'kneeling'.
        Or for example if the prompt indicates she is naked but she is wearing clothes in the image, change 'naked' to 'she removes her clothes'.
        or for example if the prompt indicates her full body is visible but the image is a close-up of her upper body, change the prompt to focus on upper body only.

        OPTION 2 IF THE INTENDED PROMPT IS NOT A GOOD FIT- Revert to a scene change transition that will create a new starting image.
        If the character's pose in the image is significantly different from what the Intended Animation Prompt describes, you should use a scene change transition prompt. 
        This is a special prompt that uses the base image of the woman, and creates a new image of her in a pose, outfit or nude, that better fits the intended animation prompt.
        **"Scene Change Transition."** - Required opener phrase for the prompt that signals the change.
        Example scene reset transition prompts:
        -- "Scene Change Transition. Image goes dark. The same woman is now kneeling on the floor, naked, looking at viewer, direct eye contact, eyes open <lora>transition</lora>."
        -- "Scene Change Transition. Image goes dark. The same woman is now standing, wearing a red lace bra and panties, looking at viewer, direct eye contact, eyes open <lora>transition</lora>."
        -- "Scene Change Transition. Image goes dark. The same woman is now laying naked on a bed, nipples and areolas visible, legs apart, looking at viewer, direct eye contact, eyes open <lora>transition</lora>."

        \n\n
        Your job is to EITHER, return the Intended Animation Prompt unchanged because the input image is a good fit, OR return a new or adjusted animation prompt. ALWAYS make sure a <lora>...</lora> tag is present at the end of the prompt.
        Remember we are trying to achieve the 'full flow of the scenes' as listed above, so keep that in mind when making adjustments that we are trying to get close to that overall flow.
        The woman's physical attributes are: {physical_attributes}. Make sure you can see these attributes in the image or you may need to use a 'scene change transition' prompt to reset her appearance.

        CRITICAL:
        RETURN ONLY THE FINAL PROMPT TO USE FOR ANIMATION, DO NOT RETURN ANY OTHER TEXT.
        Don't leave limbs or body parts in awkward positions. Adjust the prompt to have her move them back to a natural position if needed.
        Always start the prompt with 'FAST Motion.' or 'SLOW Motion.' depending on the overall motion speed indicated by the optical flow image. UNLESS we are doing a scene change transition, then you can omit the motion speed or any motion hints.
        
        """

    animation_prompt = qwen_generate(analysis_prompt, images=[optical_flow_path])

    return animation_prompt

def extract_image_tag(xml_text):
    #get and return the text inside of the <image></image> or <image_prompt></image_prompt> tags
    image_prompt = re.findall(r'<image_prompt>(.*?)</image_prompt>', xml_text, re.IGNORECASE | re.DOTALL)
    if not image_prompt:
        image_prompt = re.findall(r'<image>(.*?)</image>', xml_text, re.IGNORECASE | re.DOTALL)
    #there should only be one
    print("EXTRACTED IMAGE PROMPT: ", image_prompt)
    return image_prompt[0].strip() if image_prompt else None

def extract_scenes(xml_text):
    """
    Extract content from <scene></scene> tags and return as a list of strings.
    """
    scenes = re.findall(r'<scene>(.*?)</scene>', xml_text, re.IGNORECASE | re.DOTALL)
    return [scene.strip() for scene in scenes]

def extract_physical_description_tag(prompt):
    """
    Extracts the content within <physical_description>...</physical_description> tags from the prompt.
    Returns the extracted lora name and the prompt with the tag removed.
    """
    match = re.search(r'<physical_description>(.*?)</physical_description>', prompt, re.IGNORECASE)
    if match:
        physical_description_value = match.group(1).strip()
        clean_prompt = prompt.replace(match.group(0), "").strip()
        return physical_description_value, clean_prompt
    return None, prompt
 
def extract_lora_tag(prompt):
    """
    Extracts the content within <lora>...</lora> tags from the prompt.
    Returns the extracted lora name and the prompt with the tag removed.
    """
    match = re.search(r'<lora>(.*?)</lora>', prompt, re.IGNORECASE)
    if match:
        lora_value = match.group(1).strip()
        clean_prompt = prompt.replace(match.group(0), "").strip()
        return lora_value, clean_prompt
    return None, prompt
 

#########
#
# Video Generation Function (Comfy Server API)
#
#########
def i2v(image_path="", prompt="",folder="neon", source_image_review="", source_image_prompt="",file_name_modifier="1",lora="sex",clear_vram=False,base_file_name=None,height=None,width=None,length=81):
    global COMFY
    """
    Generate a video from an image and prompt using ComfyCall.
    """
    #create a base file name from image_path
    if base_file_name is None:
        base_file_name = os.path.basename(image_path)
    #remove file extension
    base_filename_noext = os.path.splitext(base_file_name)[0]

    result = COMFY.run(
        input_image_path=image_path,
        prompt_text=prompt,
        folder=folder,
        source_image_review=source_image_review,
        source_image_prompt=source_image_prompt,
        base_file_name=base_filename_noext,
        file_name_modifier=file_name_modifier,
        lora=lora,
        clear_vram=clear_vram,
        height=height,
        width=width,
        length=length
    )
    return result


#########
#
# Image Gen and VLM/LLM API Client Functions
#
#########
def soundfx_generate(prompt, video_path=None):
    global URL_BASE
    url = f"{URL_BASE}/soundfx"
    payload = {
        "prompt": prompt,
        "video_path": video_path,
        "negative_prompt": "music, singing, song, distortion, low quality, muffled, robotic, echo, reverb, noise, static, instrumental, background music, dialogue, talking"
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        #result is the path of the video file with sound fx added
        return response.json().get("response")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
def down_scale_image(img, target_height=360, target_width=640):
    try:
        if not os.path.exists(img):
            return img
        
        # Construct new filename
        dir_name, file_name = os.path.split(img)
        name, ext = os.path.splitext(file_name)
        new_file_name = f"{name}_{target_width}x{target_height}{ext}"
        new_file_path = os.path.join(dir_name, new_file_name)
        
        if os.path.exists(new_file_path):
            return new_file_path
        
        with Image.open(img) as image:
            resized_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            resized_image.save(new_file_path)
            
        return new_file_path
    except Exception as e:
        print(f"Error downscaling image: {e}")
        return img

def qwen_generate(prompt, images=None, unload=False, model_type='vlm',downscale_images=True):
    global URL_BASE
    
    # Process images to downscale them
    processed_images = []
    if images and downscale_images:
        for img in images:
            processed_images.append(down_scale_image(img))
            
    url = f"{URL_BASE}/generate"
    payload = {
        "prompt": prompt,
        "images": processed_images or [],
        "unload": unload,
        "model_type": model_type  # or "lm" depending on the desired model
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("response")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def image_generate(prompt,directory="image"):
    global URL_BASE
    url = f"{URL_BASE}/t2i"
    payload = {
        "prompt": prompt,
        "output_dir": directory
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("image_path")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def set_system_prompt(prompt):
    global URL_BASE
    url = f"{URL_BASE}/set_system_prompt"
    payload = {
        "system_prompt": prompt
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return True
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return False

def _load_chat_model(model_type='vlm'):
    global URL_BASE
    url = f"{URL_BASE}/load_chat"
    payload = {
        "model_type": model_type
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return True
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return False
    
def _unload_sfx_model():
    global URL_BASE
    url = f"{URL_BASE}/unload_sfx"
   
    response = requests.post(url)
    if response.status_code == 200:
        return True
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return False
    
def _unload_model():
    global URL_BASE
    url = f"{URL_BASE}/unload_chat"
   
    response = requests.post(url)
    if response.status_code == 200:
        return True
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return False
    
def _unload_image_generate():
    global URL_BASE
    url = f"{URL_BASE}/unload_t2i"
   
    response = requests.post(url)
    if response.status_code == 200:
        return True
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return False
 

if __name__ == "__main__":
    """
    HOW THIS WORKS:
    This is an all-in-one workflow that creates videos from scratch.
    FIRST - Start with an image. One can be provided, or a character description can be provided, or neither can be provided.
    -- When an image is provided, also provide source_image_prompt for logging purposes and character description if possible.
    -- When a the system generates an image, it also generates the source_image_prompt and character description.
    SECOND - An animation prompt is provided to generate a series of scene prompts.  These are used to create the 5 second animation clips that will the be stitched together to create a final video.
    THIRD - Each scene prompt is sent to the video generation function along with the image to create a series of video clips.
    -- This part of the flow will be updated. The update will be to take the last frame and evaluate it and compare it to the intended animation prompt.  IF the image will not fit well with the prompt.
    The model will be able to adjust the prompt based on the image.  OR the animation prompts will be more high level concepts, and the image analysis is generating the real detailed prompts for each clip.
    FOURTH (ish) - The video clips are sent to a video to audio model to generate sound effects for each clip.
    FIFTH - The video clips and sound effects are merged to create the final video clips with sound.

    GOAL:
    The goal is to create a fully automated workflow that generates high quality erotic videos from minimal input and oversight.  The most challenging part is that the Wan 2.2 I2V model is very sensitive to the image prompt
    so if the source image does not match the animation prompt well, it tends not to adhere to the prompt and animates what the model 'thinks' should be happening based on the image and not the prompt.
    This is why the image prompt generation and character description generation is so critical.  And why an adaptive prompt adjustment based on the image analysis is needed.
    """
    #IF using an image directly, must provide source_image_prompt for logging purposes
    #workflow_all_together(image_steering = None,shoot_folder="all_together_test5", image_gen_sys_prompt = None, animation_sys_prompt = None, animation_prompt = None, prefix = None, suffix=None, source_image_prompt="")
    image_gen_sys_prompt = """You generate detailed prompts for creating high quality, highly detailed, erotic photos of a woman character. The prompts must start with specific camera and lighting details, and end with specific shot type details.\n
    You output the full prompt inside <image_prompt></image_prompt> tags, and also output a detailed physical description of the woman inside <physical_description></physical_description> tags.
    The physical description is used later to recreate the character in other scenes, so it must match the image female feature description section exactly.
    The prompts must be suitable for generating images with stable diffusion based models. The location and pose are up to you, the clothing should be sexy and revealing, such as lingerie, miniskirts, tight dresses, yoga pants, bath robe, silk slip dress etc.
        The woman should be very sexy and attractive. facial feature, hair, accessories, eye color, eye makeup are all very important to describe.
        You also need to add a brief location, such as "a luxury hotel room", "a tropical bungalow", "an urban rooftop at sunset", "a desert cabana with a bed", "outdoors forest clearing", "nature themed spa", "a modern kitchen".
        Clothing should make sense based on the location and scene. 
        You must also be aware of the word limits on the prompt, the TOTAL prompt length must not exceed 35 words excluding the starting and ending required phrases.
        EXAMPLE OUTPUT STRUCTURE FOR A USER REQUEST:
    <image_prompt>8k, cinematic soft light, professional photography, 25 years old woman, athletic build, fair skin, long wavy brunette hair, green iris, dark mascara, pearl earrings, elegant floral sundress, standing, small breasts, upscale outdoor cafe terrace with afternoon sunlight, tables, chairs, full body view</image_prompt>
    <physical_description>25 years old woman, athletic build, fair skin, long wavy brunette hair, green iris, dark mascara, small breasts</physical_description>
    EXAMPLE OUTPUT STRUCTURE FOR NO USER REQUEST:
    <image_prompt>8k, photograph, cinema, 16mm, color graded portra 400 film, skin pores visible skin detail, remarkable detailed pupils, realistic dull skin noise short blond hair, grey-blue eyes, black choker necklace, tall, natural sagging breasts, german, wide-shot, her full-body, natural lighting </image_prompt>
    <physical_description>woman,short blond hair, grey-blue eyes, black choker necklace, tall, natural sagging breasts, german</physical_description>
    """
    
    animation_sys_prompt = """Image-to-Video Animation Director for adult content. Craft motion prompts that animate static images into video.

## CORE RULES

1. **Input image is foundation** - It contains identity, pose, scene, clothing. Prompts describe WHAT HAPPENS NEXT, not what exists.

2. **Two-phase animation** - TRANSITION first (change position), then CONTINUE (action within position). Never skip transitions.

3. **Face visibility is CRITICAL** - Woman's face MUST be visible with eye contact for character consistency. If face won't be visible (e.g., doggy from behind), end with a transition to reestablish face visibility.

4. **Track Nudity** - Pay attention if clothing is removed at some point, explicitly state resulting naked body parts visible (e.g., "bra removed, breasts visible") in all subsequent scene transitions.

5. **LORA tags** - Always end prompts with appropriate <lora>...</lora> tag to guide animation style.

## TRANSITION FORMAT
```
Scene Change Transition. [Image goes dark.] Then we show the same woman [naked/clothing state]. [NEW POSITION]. [CAMERA]. [EYE CONTACT].
```

Key phrases:
- "Scene Change Transition." - Required opener for position changes
- "Image goes dark." - Use when introducing male partner
- "the same woman" - Maintains identity
- "POV" - Viewer becomes participant
- Always state if naked/nude/clothing removed

## CONTINUATION FORMAT
```
[ACTION]. [MOTION DETAILS]. direct gaze at viewer, eyes open.
```

Be specific: "goes in and out", "pulls backwards", "very slowly"

## CAMERA VOCABULARY
dolly-in/out, orbit, tilt-up/down, wide shot, high angle, POV

## SEQUENCE STRUCTURE
OPENING → TRANSITION (new pose) → CONTINUE (action) → CONTINUE → TRANSITION (new pose) → CONTINUE → CONTINUE (climax) → CONTINUE (reaction)

## CHECKLIST
- Transition language for position changes?
- Motion/action described (not static appearance)?
- Face visible with eye contact specified?
- Camera movement included?
- Subject identity maintained?

## BE CREATIVE
- User will give you ideas and concepts. Adapt them into dynamic animation prompts. Stay to the overall template structure and transition wording but be imaginative within those rules and scene actions.
Create engaging, explicit, erotic, arrousing, porn animations that bring static images to life while adhering to these guidelines."""
    
    animation_prompt = f"""I have provided you with an image of a woman. You are to create a series of detailed scene descriptions for an erotic photoshoot or video shoot featuring this same woman
        
    # Scenes progression should ALWAYS use these EXACT phrases when getting ready for a new type of scene:
    TRANSITION TO SIDE VIEW (pre DOGGY STYLE SEX):
        'Scene Change Transition. The same woman, [her clothing is gone she is naked] on her hands and knees, side view looking back over shoulder, face visible, eye contact. wide-shot'

    TRANSITION TO LYING DOWN (pre MISSIONARY SEX)
        'Scene Change Transition. Then we show the same woman laying down. high angle. eye contact with viewer. POV'

    TRANSITION TO KNEELING (pre BLOWJOB)
        'Scene Change Transition. Then we show the same woman kneeling down, both shoulders in view. high angle wide view. eye contact with viewer, face looking at viewer. POV'
    
    TRANSITION TO STANDING (pre STRIPTEASE)
        'Scene Change Transition. Then we show the same woman standing up. eye contact with viewer, face looking at viewer.'

    [flirty motions, laughing teasing motions, smiling, play with hair, adjust clothing, etc] Animation Examples:
    "she does a half turn, arches back showing ass, then turns back facing forward, smile. direct gaze at viewer, eyes open. <lora>standard</lora>"
    "she smiles, her hand slowly moves down her thigh sensually. direct gaze at viewer, eyes open. <lora>standard</lora>"
    "she brushes her hair with her hand, laughs playfully. direct gaze at viewer, eyes open.<lora>standard</lora>"
    "she adjusts her top, smerks. direct gaze at viewer, smile, eyes open. <lora>standard</lora>"
    "she does a half turn and spanks her ass then turns back, smiles. direct gaze at viewer, smile, eyes open. <lora>standard</lora>"
    "she lifts her hand to her mouth, then blows a kiss at viewer. direct gaze at viewer, smile, eyes open. <lora>standard</lora>"
    "she runs her hand through her hair, smiles playfully. direct gaze at viewer, eyes open.<lora>standard</lora>"
    "she adjusts her top, her hands pushing her breasts together, clevage. direct gaze at viewer, smile, eyes open. <lora>standard</lora>"
    "she extends her hand towards viewer, motions with index finger to come and approach with a forward and backward motion of the finger, smiles and puts her arm back down.<lora>standard</lora>"
    "she adjusts her top, pushes breasts together, cleavage visible. direct gaze at viewer, smile, eyes open. <lora>standard</lora>"
    "she puts her index finger to her lips vertically making a 'shhhh' motion, then smiles. intense stare at viewer <lora>standard</lora>"
    
    NEVER use more than 3 of these types of flirty motion scenes in a row when starting a sequence of scenes.

    SCENE TYPE TEMPLATES (choose a concept on your creative direction. You DO NOT need to be limited to these exact scenes, but you MUST follow the overall structure and wording style for transitions and actions):

    ##STRIPTEASE##
    "provocative walk towards viewer. blink, eye contact with viewer. dolly-out<lora>standard</lora>"
    "she does a half turn, arches back showing ass, then turns back facing forward. direct gaze at viewer, eyes open. <lora>standard</lora>"
    "she smiles, her hand slowly moves down her thigh sensually. direct gaze at viewer, eyes open. <lora>standard</lora>"
    "she brushes her hair with her hand, laughs playfully. direct gaze at viewer, eyes open.<lora>standard</lora>"
    "she adjusts her top, smerks. direct gaze at viewer, smile, eyes open. <lora>standard</lora>"

    Transition to nude
    First transition standing: 'Scene Change Transition. Then we show the same woman provocative sexual pose standing, wide shot, eye contact with viewer. <lora>transition</lora>'
    Then transition to no top: 'The woman quickly lifting her [bra/shirt/top] over her head and off, discarding to the floor<lora>standard</lora>'

    'She bends forward as she pulls her [panties/pants/skirt/bottoms] down. eye contact with viewer. then stands up straight again <lora>standard</lora>'
    'she stands fully nude, hands on hips, posing seductively. eye contact with viewer. smile <lora>standard</lora>
    'she does a slow spin, showing her naked body from all angles. eye contact with viewer. <lora>standard</lora>
    'she places her hands on her breasts, squeezes them gently. facial expression seduction, face visible, eye contact with viewer. <lora>standard</lora>'
    
    ##MISSIONARY##
    Opening scene examples, use 3 the same as or similar:
    "provocative walk towards viewer. blink, eye contact with viewer. dolly-out<lora>standard</lora>"
    "she does a half turn and spanks her ass then turns back. direct gaze at viewer, smile, eyes open. <lora>standard</lora>"
    "she lifts her hand to her mouth, then blows a kiss at viewer. direct gaze at viewer, smile, eyes open. <lora>standard</lora>"
    
    First transition to laying down: 'Scene Change Transition. Then we show the same woman laying down she is naked. fully nude no clothing, vagina and breasts visible. high angle. eye contact with viewer. POV. <lora>transition</lora>'
    Then she gets naked (if she has clothing on)
    "her arms raise and quickly removes clothing top over head, throws clothing to side, revealing naked breasts. direct gaze at viewer, eyes open. full body view.<lora>standard</lora>"
    "she lowers her arms and removes clothing bottom down legs, revealing naked vagina. direct gaze at viewer, eyes open. full body view.<lora>standard</lora>"
    Then transition to act of sex: 'Then we show the same woman, her body is visible, one leg raised knee bent, direct gaze with viewer, eyes open. A mans erect penis, testicles, his penis entering her vagina, thrust in to her, POV missionary sex, vagina visible. viewer is looking down at her. <lora>sex</lora>'

    Then continue with any 2 of these:
    1. "sexual intercourse, she is enoying herself. His penis thrusts in to her vagina. <lora>sex</lora>"
    2. "sexual intercourse, her hands grab the back of her knees and lift her legs. direct gaze at viewer, blinks then eyes open. <lora>sex</lora>"
    3. "sexual intercourse, she raises her finger to her mouth and sucks on it, expression of joy. <lora>sex</lora>"
    4. "sexual intercourse, she raises her hands to her breasts and squeezes them with expression of delight. <lora>sex</lora>"

    Then finish with this ejaculation scene
    "His hips move back all the way and the penis exits vagina, he stops moving. penis is visible and above her vagina ejaculating on her stomach. she smiles, direct gaze with viewer, her eyes open. <lora>ejaculation</lora>"

    Then a final scene from the list or a similar one:
    "She wipes the cum off her stomach, puts her finger in her mouth and licks it off<lora>standard</lora>"
    "She wipes the cum off her stomach, rubs it on her breasts <lora>standard</lora>"
    "She laughs and rubs the cum around on her stomach<lora>standard</lora>"

    ##BLOWJOB##
    Opening scene examples, use 3 the same as or similar:
    "provocative walk towards viewer. blink, eye contact with viewer. dolly-out<lora>standard</lora>"
    "she brushes her hair with her hand, laughs playfully. direct gaze at viewer, eyes open.<lora>standard</lora>"
    "she adjusts her top, her hands pushing her breasts together, clevage. direct gaze at viewer, smile, eyes open. <lora>standard</lora>"
    
    First transition to kneeling down: 'Scene Change Transition. Then we show the same woman kneeling down, both shoulders in view. high angle wide view. eye contact with viewer, face looking at viewer. POV. <lora>transition</lora>'
    Then she removes her top/shirt/bra: "her arms quickly remove clothing top over head, revealing nude breasts and nipples. direct gaze at viewer, eyes open. full body view.<lora>standard</lora>"
    Then transition to the act of blowjob: 'The same woman on her knees puts his large erect penis in her mouth. POV. her face is visible, direct gaze with viewer, eyes open. She is putting a mans penis in her mouth. viewer is looking down at her. <lora>blowjob</lora>'

    Then continue with any of these
    "blowjob, penis head and shaft goes in and out of her mouth. direct gaze at viewer, slow blink, eyes open. <lora>blowjob</lora>"
    "blowjob, penis head comes out of her mouth she takes a breath then continues giving the blowjob. direct gaze at viewer, eyes open. <lora>blowjob</lora>"
    "blowjob, she slows down doing slow rythmic motion back and forth on the penis shaft. <lora>blowjob</lora>"

    Then finish with this ejaculation scene
    Then ejaculation: "She pulls her head back and penis exits her mouth. Her mouth is open, man's erect penis starts ejaculating cum in her mouth. she flinches with surprise when cum enters her mouth, her eyes get wide, eyebrows raise. POV viewer high angle, eye contact with viewer. her face is visible, direct gaze with viewer, eyes open. <lora>ejaculation</lora>"

    Then 1 final scene from the list or a similar one to continue the ending:
    "facial. she smiles raises her hand to her face, wipes the cum with her finger and sucks her finger with her mouth.<lora>standard<lora>"
    "facial. she laughs and wipes cum out of eyes and off her face with her hands, direct gaze at viewer eyes open.<lora>standard<lora>"
    "facial. she leans back and rubs the cum off her face and on to her breasts direct, eye contact with viewer.<lora>standard<lora>"



    ##DOGGY STYLE##
    Opening scene examples, use 3 of the below or similar starting scenes:
    "provocative walk towards viewer. blink, eye contact with viewer. dolly-out<lora>standard</lora>"
    "she extends her hand towards viewer, motions with index finger to come and approach with a forward and backward motion of the finger, smiles and puts her arm back down.<lora>standard</lora>"
    "she adjusts her top, pushes breasts together, cleavage visible. direct gaze at viewer, smile, eyes open. <lora>standard</lora>"
    
    First transition to rear view : 'Scene Change Transition. Then we show the same woman from the side. turns and looks back over her shoulder, completly naked, face visible. eye contact with viewer, eyes open, wide shot<lora>transition</lora>'
    Then she removes pants/skirt/bottoms: "turns and bends over, removes clothing bottom down legs, revealing naked ass.<lora>standard</lora>"
    Then transition to act of sex: 'A Man with erect penis is behind her ass, he puts his hands on her hips, then thrusts his penis in her vagina having doggy style sex, thrusting his penis in and out of her vagina, her breasts swey. she is looking back over her shoulder, her face visible.<lora>doggystyle</lora>'

    Then continue with any 2 of these:
    1. "doggy style intercourse, she is enoying herself. His penis thrusts in to her vagina. looking back over shoulder, eye contact, eyes open<lora>sex</lora>"
    2. "doggy style intercourse, with one hand reach back and grab her ass. looking back over shoulder, eye contact, eyes open<lora>sex</lora>"
    3. "doggy style intercourse, She extends her arms forward and arches her back. looking back over shoulder, eye contact, eyes open<lora>sex</lora>"
    4. "doggy style intercourse, with one arm She grabs her breasts, looking back over shoulder, eye contact, eyes open<lora>sex</lora>"

    Then finish with this ejaculation scene
    "The sex stops. the man's hips move backwards and his penis pulls all the way out of her vagian, he holds his penis and ejaculates cum on to her ass and back. she is looking back over shoulder direct gaze with viewer, her eyes open. <lora>ejaculation</lora>"

    Then 1 final scene from the list or a similar one:
    "She reaches back and wipes the cum off her ass, puts her finger in her mouth and licks it off, eye contact with viewer<lora>standard</lora>"
    "She reaches back rubs cum on her ass, direct gaze with viewer, eyes open <lora>standard</lora>"
    "She smiles and sits upright, eye contact with viewer, direct gaze, dolly-in face close up<lora>standard</lora>"


    EXAMPLE OUTPUT STRUCTURE FOR A MISSIONARY SCENE- DOGGY STYLE, BLOWJOB and STRIPTEASE would have a similar flow structure but different prompts:
    <image>8k, cinematic soft light, professional photography, 25 years old woman, athletic build, fair skin, long wavy brunette hair, green iris, dark mascara, pearl earrings, elegant floral sundress, standing, small breasts, upscale outdoor cafe terrace with afternoon sunlight, tables, chairs, full body view</image>
    <physical_description>25 years old woman, athletic build, fair skin, long wavy brunette hair, green iris, dark mascara, small breasts</physical_description>

    <scene>she does a provocative walk towards viewer....<lora>standard</lora></scene>
    <scene>..she brushes her hair with her hand. direct gaze at viewer, ..<lora>standard</lora></scene>
    <scene>...she adjusts her top, cleavage visible...<lora>standard</lora></scene>
    <scene>Scene Change Transition. Then we show the same woman laying down. fully nude no clothing, vagina and breasts visible. high angle. eye contact with viewer. POV. <lora>transition</lora></scene>
    <scene>she removes clothing top slowly casting it aside, revealing naked breasts. direct gaze at viewer, eyes open. full body view.<lora>standard</lora></scene>
    <scene>she removes clothing bottom slowly casting it aside, revealing naked vagina. direct gaze at viewer, eyes open. full body view.<lora>standard</lora></scene>
    <scene>Then we show the same woman naked on her back, her body is visible, one leg raised knee bent, direct gaze with viewer, eyes open. A mans large penis entering her vagina, thrusting his hips in to her, they have missionary sex, vagina visible. viewer is looking down at her. <lora>sex</lora></scene>
    <scene>.. intercourse, she is enoying herself...<lora>sex</lora></scene>
    <scene>... intercourse, she raises her hands to her breasts ... <lora>sex</lora></scene>
    <scene>Then sex motion stops, his hips move back all the way and the penis exits vagina, penis is visible and above her ejaculating cum on her stomach and breasts. direct gaze with viewer, her eyes open. <lora>ejaculation</lora></scene>
    <scene>She wipes the cum off her stomach, rubs it on her breasts<lora>standard</lora></scene>
    
    # I will give you a female character image that has a location pick a scene type (striptease, missionary, doggy style, blowjob)
    # Create a sequence of about 10 scenes involving the female character that will be combined into a single video.

    CRITICAL RULES FOR ANIMATION SCENES:
    - Maximum 35 words per scene.
    - The Animation model already sees the image, it does not need images details described. 
    - DO NOT state the full location description in each scene, only specifics about objects if they are relevant (like a bed in the cabana)
    - Follow close to the wording in the examples
    - Scene transitions with the transition tag must include description of what if any clothing is removed or was adjusted to prepare for the next scene.
    - you MUST use the exact phrases for transitions in the examples. When a scene changes we need to transition the woman's pose based on what the next scene requires.
    - MUST indicate the result of clothing removal being 'naked' or 'nude breasts and nipples visible' or 'vagina visible' as appropriate for the scene.
    - ALWAYS restate that she is nude or naked at the end of subsequent prompts after scene transitions and undressing are used to help with nudity consistency
    - NEVER use a <lora>transition</lora> with successive scenes. Those tags must ALWAYS be separated by at least one different scene with a different lora tag
    Transitions are needed before:
    ---transition to kneeling for for blowjob scenes
    ---transition to lying on back for missionary sex
    ---transition to provocative standing or other poses for striptease scene changes
    ---transition to side view on hands and knees, she is looking back over shoulder for doggy style sex
    
    - About 10 scenes
    - Must MUST have a LORA tag at the end of the prompt, inside of the scene tag, to help the animation. tag choices are:
    -- <lora>standard</lora> for general movements, striptease scenes
    -- <lora>masturbation</lora> for masturbation scenes
    -- <lora>sex</lora> for any missionary or side type intercourse sexual act scenes
    -- <lora>doggystyle</lora> for doggy style intercourse sexual act scenes
    -- <lora>blowjob</lora> for blowjob scenes   
    -- <lora>ejaculation</lora> for any or ejaculation scenes (facial, on body, on breasts, etc) this is used at the end typically
    -- <lora>transition</lora> for when the scene transitions to a new pose in anticipation of a new scene

    SCENARIO CONCEPTS YOU SHOULD BUILD AROUND:
        FLIRTATION TO BLOWJOB
            - provocotive movements, [flirty facial expressions, flirty motions, laughing teasing motions, smiling, play with hair, adjust clothing, etc], kneeling, remove top/shirt, blowjob, ejaculation
        FLIRTATION TO SEX
            - provocotive movements,[flirty facial expressionsflirty motions, laughing teasing motions, smiling, play with hair, adjust clothing, etc], remove clothing, sex, ejaculation
        FLIRTATION TO DOGGY STYLE
            - provocotive movements,[flirty facial expressionsflirty motions, laughing teasing motions, smiling, play with hair, adjust clothing, etc], remove clothing, sex, ejaculation
        FLIRTATION TO STRIPTEASE
            - provocotive movements, [flirty facial expressionsflirty motions, laughing teasing motions, smiling, play with hair, adjust clothing, etc], removes top/bra/shirt plays with nipples and shows off breasts, removes bottoms/panties/skirt shows ass and vagina, sexual poses

    Use the provided starting image: and come up with a good sexual shoot flow with about 10 animation scenes following the progression structure above.
    Be creative but stick to the structure.
    ALWAYS have ejaculation lora near the end
    Be creative with how the porno scene progresses. don't use more than 3 [flirty motions, laughing teasing motions, smiling, play with hair, adjust clothing, etc] scenes in the begining
    
    YOU MUST use the exact phrase 'Scene Change Transition...' for transition changes and sexual act starts
    Output the structure about 10 <scene>...</scene> tags.
    """

    prefix = "8k, photograph, color graded portra 400 film, remarkable color, ultra realistic, skin pores visible, remarkable detailed pupils, realistic dull skin noise"
    suffix = "full-body view, cinematic soft light"


    image_steering = {'diana':'wavy brown hair, hazel eyes, gold hoop earrings, young spanish girl, wearing a dark blue with white lace leotard lingerie',
                      'elena':'face framing black hair, brown eyes, curvy, arabian',
                      'fiona':'short blond hair, grey-blue eyes, black choker necklace, tall, natural sagging breasts, german',
                      'cindy':'auburn hair, green eyes, light freckles, red lingerie, irish',
                      'alison':'blond ponytail, brown eyes, thin eyebrows, long eyelashes, glasses, small breasts, french',
                      'beth':'flowing brunette hair, blue eyes, small sagging breasts, british'}
    
    _ = _unload_sfx_model()
    _ = _unload_image_generate()
    _ = COMFY.aggressive_cleanup()
    for x in range(0, 2):
        start_time = time.time()
        #key = f"{list(image_steering.keys())[x]}"#female model name
        key = f"fullAuto{x+1}"#female model name
        print(f"STARTING WORKFLOW '{key}'")
        #end = workflow_all_together(image_steering = image_steering[list(image_steering.keys())[x]],shoot_folder=f"jan21_{key}", image_gen_sys_prompt = image_gen_sys_prompt, animation_sys_prompt = animation_sys_prompt, animation_prompt = animation_prompt, prefix = prefix, suffix=suffix, source_image_prompt="")
        end = workflow_all_together(shoot_folder=f"jan21_{key}", image_gen_sys_prompt = image_gen_sys_prompt, animation_sys_prompt = animation_sys_prompt, animation_prompt = animation_prompt, prefix = prefix, suffix=suffix, source_image_prompt="")

        _ = _unload_sfx_model()
        time.sleep(10)#give time to clear vram
        _ = _unload_image_generate()
        time.sleep(10)#give time to clear vram
        _ = _unload_model()
        end_time = time.time()
        print(f"WORKFLOW '{key}' TOTAL TIME: {end_time - start_time} seconds")
        time.sleep(10)#give time to clear vram