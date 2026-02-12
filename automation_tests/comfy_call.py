import json
import time
import urllib.request
import urllib.parse
import requests
import websocket
import uuid
import random
import os
import gc

#TODO: unload models after inference to free up VRAM when implementing Mixture of LORA approach
## After your inference - assuming that is where COMFY_SERVER is defined
#requests.post('http://localhost:8188/free', json={'unload_models': True})
#

class ComfyCall:
    # ================= CONFIGURATION =================
    COMFY_SERVER = "127.0.0.1:8000" # Local ComfyUI address
    WORKFLOW_FILE = "video_wan2_2_REMIX_14B_i2v_nsfwLORAs_API.json" # The file you saved via "Export (API)"
    
    # IDS FROM COMFYUI JSON
    IMAGE_NODE_ID = "97"
    PROMPT_NODE_ID = "93"
    NEGATIVE_PROMPT_NODE_ID = "89"
    SEED_NODE_ID = "86"
    IMAGE_INPUT_NODE_ID = "98"
    FINAL_FRAME_SELECTOR_NODE_ID = "119"
    #lower resolution for faster processing
    IMAGE_INPUT_HEIGHT = 832 #640 1024
    IMAGE_INPUT_WIDTH = 832 #640 1024

    #LORA SETTINGS:
    #dynamically set these based on desired content, .7 high, 1 for low and 1 on clip.strength. Set the others to 0. Only BJ and ejaculation need their own. Sex can be general.
    DOGGY_HIGH = 0.5
    DOGGY_LOW = 0.5
    DOGGY_STRENGTH = 1.0

    EJ_HIGH = 0.6
    EJ_LOW = 0.6
    EJ_STRENGTH = 1.0

    BJ_HIGH = 0.7
    BJ_LOW = 0.7
    BJ_STRENGTH = 1.0

    MBATE_HIGH = 0.2
    MBATE_LOW = 0.2
    MBATE_STRENGTH = 1.0

    LEG_LIFTED_HIGH = 0.2
    LEG_LIFTED_LOW = 0.2
    LEG_LIFTED_STRENGTH = 1.0

    
    OUTPUT_DIR = "C:\\Users\\x\\Documents\\code\\local_jarvis\\xserver\\autogen\\"
    # =================================================

    def __init__(self):
        self.client_id = str(uuid.uuid4())
        self.ws = None
        print(f"ComfyCall client initialized with Client ID: {self.client_id}")
        self.connect()
        print("ComfyCall WebSocket connection established.")

    def connect(self):
        """Connects to the ComfyUI WebSocket"""
        self.ws = websocket.WebSocket()
        self.ws.connect(f"ws://{self.COMFY_SERVER}/ws?clientId={self.client_id}")

    def ensure_connected(self):
        """Ensures the WebSocket is connected, reconnecting if necessary"""
        try:
            if self.ws is None or not self.ws.connected:
                print("WebSocket not connected, reconnecting...")
                self.connect()
                print("WebSocket reconnected.")
            else:
                # Connection appears valid, but let's flush any stale messages
                self._flush_stale_messages()
        except Exception as e:
            print(f"Connection check failed ({e}), forcing reconnect...")
            self.connect()
            print("WebSocket reconnected.")

    def _flush_stale_messages(self):
            """Flush any stale messages from the WebSocket before starting new work"""
            self.ws.settimeout(0.1)  # Non-blocking
            flushed_count = 0
            try:
                while True:
                    try:
                        msg = self.ws.recv()
                        flushed_count += 1
                    except websocket.WebSocketTimeoutException:
                        break  # No more messages
            except Exception:
                pass  # Ignore errors during flush
            finally:
                self.ws.settimeout(None)  # Reset to blocking
            if flushed_count > 0:
                print(f"DEBUG: Flushed {flushed_count} stale WebSocket messages")

    def upload_image(self, file_path):
        """Uploads the local image to ComfyUI"""
        with open(file_path, "rb") as file:
            files = {"image": file}
            data = {"overwrite": "true"}
            response = requests.post(f"http://{self.COMFY_SERVER}/upload/image", files=files, data=data)
        return response.json()

    def queue_prompt(self, workflow):
        """Sends the workflow to the queue"""
        p = {"prompt": workflow, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.COMFY_SERVER}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def track_progress(self, prompt_id):
        """Listens to WebSocket for completion"""
        while True:
            out = self.ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        print("Execution complete.")
                        break
            else:
                continue
        return

    def get_history(self, prompt_id):
        """Fetches the final results (filenames)"""
        with urllib.request.urlopen(f"http://{self.COMFY_SERVER}/history/{prompt_id}") as response:
            return json.loads(response.read())
    
    def download_file(self, filename, subfolder, folder_type):
        """Downloads the generated video/image"""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        query = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.COMFY_SERVER}/view?{query}") as response:
            return response.read()

    def close(self):
        """Closes the WebSocket connection"""
        if self.ws:
            self.ws.close()

    def aggressive_cleanup(self):
        """
        Aggressively cleans up ComfyUI's VRAM caches.
        This is the KEY to solving your OOM issue with LoRA swapping.
        """
        print("\n🧹 Performing aggressive VRAM cleanup...")
        
        try:
            # 1. Free unneeded memory (this is better than just /free)
            free_response = requests.post(
                f"http://{self.COMFY_SERVER}/free",
                json={"unload_models": True, "free_memory": True}
            )
            print(f"   ✓ /free endpoint: {free_response.status_code}")
            
            # 2. Clear the queue (removes any pending jobs)
            queue_response = requests.post(
                f"http://{self.COMFY_SERVER}/queue",
                json={"clear": True}
            )
            print(f"   ✓ Queue cleared: {queue_response.status_code}")
            
            # 3. Interrupt any running execution
            interrupt_response = requests.post(
                f"http://{self.COMFY_SERVER}/interrupt"
            )
            print(f"   ✓ Interrupted: {interrupt_response.status_code}")
            
            # 4. Give ComfyUI time to actually clean up
            time.sleep(2)
            
            # 5. Trigger Python garbage collection locally
            gc.collect()
            
            print("   ✓ Cleanup complete\n")

            return True
            
        except Exception as e:
            print(f"   ⚠ Cleanup warning (non-fatal): {e}")

            return False

    def edit_image(self, input_image_path, prompt_text, base_file_name="base_file_name", folder="images", file_name_modifier="", source_image_review="", source_image_prompt=""):
        """
        Executes a workflow to edit an image based on a prompt.
        Returns a dictionary of saved file paths matching the format of run().
        Structure of returned saved_file paths:
                {'image_final': 'path_to_edited_image',
                 'edit_prompt': 'path_to_edit_prompt.txt',
                 'source_image_prompt': 'path_to_source_image_prompt.txt',
                 'source_image_review': 'path_to_source_image_review.txt',
                 'image_start': 'path_to_input_image'}
        """
        EDIT_WORKFLOW_FILE = "qwenImgEdit2511.json" # The file you saved via "Export (API)" for image editing
        image_node_id = "41"
        prompt_node_id = "105"
        # Ensure WebSocket is connected
        self.ensure_connected()

        # Upload Your Image
        print(f"Uploading {input_image_path}...")
        upload_resp = self.upload_image(input_image_path)
        server_filename = upload_resp["name"] # The name ComfyUI assigned to it
        print(f"Uploaded as: {server_filename}")

        # Load and Modify Workflow
        with open(EDIT_WORKFLOW_FILE, "r") as f:
            workflow = json.load(f)
        
        # Update Image Node
        workflow[image_node_id]["inputs"]["image"] = server_filename

        # Update Prompt Node
        workflow[prompt_node_id]["inputs"]["prompt"] = prompt_text

        # Queue the Workflow
        response = self.queue_prompt(workflow)
        prompt_id = response['prompt_id']
        print(f"Comfy job queued! Prompt ID: {prompt_id}")

        # Wait for completion
        self.track_progress(prompt_id)  # Wait for job to complete

        # Download Result
        history = self.get_history(prompt_id)[prompt_id]
        saved_files = {}
        
        output_dir = os.path.join(self.OUTPUT_DIR, folder)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ts = int(time.time())
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            
            # Check for images
            output_data = []
            if 'images' in node_output: output_data += node_output['images']

            for item in output_data:
                print(f"Downloading {item['filename']}...")
                file_data = self.download_file(item['filename'], item['subfolder'], item['type'])
                
                ext = os.path.splitext(item['filename'])[1]
                new_filename = f"{base_file_name}_{file_name_modifier}_edited{ext}"
                file_key = "image_final"
                
                output_path = os.path.join(output_dir, new_filename)
                with open(output_path, "wb") as f:
                    f.write(file_data)

                print(f"Saved to {output_path}")
                saved_files[file_key] = output_path

            # Save the edit prompt, source image prompt, and source image review to text files for reference
            edit_prompt_path = os.path.join(output_dir, f"{base_file_name}_{file_name_modifier}_edit_prompt.txt")
            with open(edit_prompt_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)
            
            source_img_path = os.path.join(output_dir, f"{base_file_name}_{file_name_modifier}_source_image_prompt.txt")
            with open(source_img_path, "w", encoding="utf-8") as f:
                f.write(source_image_prompt)

            source_review_path = os.path.join(output_dir, f"{base_file_name}_{file_name_modifier}_source_image_review.txt")
            with open(source_review_path, "w", encoding="utf-8") as f:
                f.write(source_image_review)

            saved_files["edit_prompt"] = edit_prompt_path
            saved_files["source_image_prompt"] = source_img_path
            saved_files["source_image_review"] = source_review_path
            saved_files["image_start"] = input_image_path

        return saved_files

    #def run(self, input_image_path, prompt_text, folder="movies"):
    def run(self, input_image_path, prompt_text, base_file_name="base_file_name",folder="movies",source_image_review="",source_image_prompt="",file_name_modifier="",lora="sex",clear_vram=False,height=None,width=None,length=81):
        """
        length is the total number of frames to generate, default 81 (16fps for 5 seconds)
        Executes the full workflow:
        1. Connects to WebSocket
        2. Uploads image
        3. Modifies workflow with image and prompt
        4. Queues job
        5. Waits for completion
        6. Downloads results

        returns a dictionary of saved file paths.
        Structure of returned saved_file paths:
                {'video': 'path_to_video',
                 'image_final': 'path_to_final_image',
                 'animation_prompt': 'path_to_animation_prompt.txt',
                 'source_image_prompt': 'path_to_source_image_prompt.txt',
                 'source_image_review': 'path_to_source_image_review.txt',
                 'image_start': 'path_to_input_image'}
        """
        #add timeout to ws connect
        # self.ws.settimeout(300)  # Set timeout to 300 seconds, usually takes 3 minutes for video, this should be enough
        try:
            # 1. Ensure WebSocket is connected (reconnect if needed)
            self.ensure_connected()

            # 2. Upload Your Image
            print(f"Uploading {input_image_path}...")
            upload_resp = self.upload_image(input_image_path)
            server_filename = upload_resp["name"] # The name ComfyUI assigned to it
            print(f"Uploaded as: {server_filename}")

            # 3. Load and Modify Workflow
            # Assuming the workflow file is in the same directory or path is correct
            """if not os.path.exists(self.WORKFLOW_FILE):
                 # Try to find it relative to this file if not found
                 current_dir = os.path.dirname(os.path.abspath(__file__))
                 workflow_path = os.path.join(current_dir, self.WORKFLOW_FILE)
            else:
                workflow_path = self.WORKFLOW_FILE

            with open(workflow_path, "r") as f:"""

            #DYNAMICAlly LOAD DIFFERENT WORKFLOWS BASED ON LORA TYPE
            #if lora == "sex":
            #    self.WORKFLOW_FILE = "video_wan2_2_14B_i2v_nsfw(sex)_API.json"
            #elif lora == "ejaculation":
            #    self.WORKFLOW_FILE = "video_wan2_2_14B_i2v_nsfw(facial)_API.json"
            #elif lora == "blowjob":
            #    self.WORKFLOW_FILE = "video_wan2_2_14B_i2v_nsfw(blowjob)_API.json"
            with open(self.WORKFLOW_FILE, "r") as f:
                workflow = json.load(f)
            
            # Update Image Node
            workflow[self.IMAGE_NODE_ID]["inputs"]["image"] = server_filename

            # Update Prompt Node
            workflow[self.PROMPT_NODE_ID]["inputs"]["text"] = prompt_text

            # Update Negative Prompt Node
            workflow[self.NEGATIVE_PROMPT_NODE_ID]["inputs"]["text"] = "gay, transvestite, hands, chewing, biting, eating, messy, distorted, deformed, disfigured, ugly, tiling, poorly drawn, mutation, mutated, extra limbs, cloned face, disfigured, out of frame, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, fused fingers, too many fingers, long neck"
            
            # Update Image Input Node (for resizing)
            if height is None:
                height = self.IMAGE_INPUT_HEIGHT
            if width is None:
                width = self.IMAGE_INPUT_WIDTH
            workflow[self.IMAGE_INPUT_NODE_ID]["inputs"]["width"] = width
            workflow[self.IMAGE_INPUT_NODE_ID]["inputs"]["height"] = height
            workflow[self.IMAGE_INPUT_NODE_ID]["inputs"]["length"] = length
            #also set the final frame selector to be the same as length
            workflow[self.FINAL_FRAME_SELECTOR_NODE_ID]["inputs"]["batch_index"] = length
 
            #COMMENT OUT TO TRY AND USE A SINGLE JSON, NO DYNAMIC LORA
            #RESET ALL LORA VALUES TO 0
            #NSFW
            workflow["147"]["inputs"]["strength_model"] = 0
            workflow["149"]["inputs"]["strength_model"] = 0
            workflow["149"]["inputs"]["strength_clip"] = 1
            #BJ
            workflow["152"]["inputs"]["strength_model"] = 0
            workflow["153"]["inputs"]["strength_model"] = 0
            workflow["153"]["inputs"]["strength_clip"] = 1
            #FACIAL
            workflow["155"]["inputs"]["strength_model"] = 0
            workflow["154"]["inputs"]["strength_model"] = 0
            workflow["154"]["inputs"]["strength_clip"] = 1
            #OTHER (New Nodes)
            workflow["156"]["inputs"]["strength_model"] = 0 #leg lift spoon
            workflow["158"]["inputs"]["strength_model"] = 0 #leg lift spoon
            workflow["158"]["inputs"]["strength_clip"] = 1 #leg lift spoon
            workflow["157"]["inputs"]["strength_model"] = 0 #female masturbation
            workflow["159"]["inputs"]["strength_model"] = 0 #female masturbation
            workflow["159"]["inputs"]["strength_clip"] = 1 #female masturbation
            
            #Set LORA VALUES
            if lora == "standard":
                pass  # All LORA values are already set to 0
                #workflow["157"]["inputs"]["strength_model"] = self.MBATE_HIGH #female masturbation
                #workflow["159"]["inputs"]["strength_model"] = self.MBATE_LOW #female masturbation
                #workflow["159"]["inputs"]["strength_clip"] = self.MBATE_STRENGTH #female masturbation
                #workflow["147"]["inputs"]["strength_model"] = 0.2 #Doggy style
                #workflow["149"]["inputs"]["strength_model"] = 0.2 #Doggy style
                #workflow["149"]["inputs"]["strength_clip"] = 1 #Doggy style
            elif lora == "ejaculation":
                workflow["155"]["inputs"]["strength_model"] = self.EJ_HIGH
                workflow["154"]["inputs"]["strength_model"] = self.EJ_LOW
                workflow["154"]["inputs"]["strength_clip"] = self.EJ_STRENGTH
                workflow["152"]["inputs"]["strength_model"] = 0.2 #bj lora
                workflow["153"]["inputs"]["strength_model"] = 0.2 #bj lora
                workflow["153"]["inputs"]["strength_clip"] = 1 #bj lora
            elif lora == "blowjob":
                workflow["152"]["inputs"]["strength_model"] = self.BJ_HIGH
                workflow["153"]["inputs"]["strength_model"] = self.BJ_LOW
                workflow["153"]["inputs"]["strength_clip"] = self.BJ_STRENGTH
            elif lora == "sex":
                workflow["156"]["inputs"]["strength_model"] = self.LEG_LIFTED_HIGH #leg lift spoon
                workflow["158"]["inputs"]["strength_model"] = self.LEG_LIFTED_LOW #leg lift spoon
                workflow["158"]["inputs"]["strength_clip"] = self.LEG_LIFTED_STRENGTH #leg lift spoon
            elif lora == "masturbation":
                workflow["157"]["inputs"]["strength_model"] = self.MBATE_HIGH #female masturbation
                workflow["159"]["inputs"]["strength_model"] = self.MBATE_LOW #female masturbation
                workflow["159"]["inputs"]["strength_clip"] = self.MBATE_STRENGTH #female masturbation
            elif lora == "doggystyle":
                workflow["147"]["inputs"]["strength_model"] = self.DOGGY_HIGH #Doggy style
                workflow["149"]["inputs"]["strength_model"] = self.DOGGY_LOW #Doggy style
                workflow["149"]["inputs"]["strength_clip"] = self.DOGGY_STRENGTH #Doggy style
            

            # Randomize Seed (so you don't get the exact same video every time)
            if self.SEED_NODE_ID in workflow:

                #WAS USING A STATIC SEED FOR TESTING PURPOSES, SWITCHING BACK TO RANDOM
                workflow[self.SEED_NODE_ID]["inputs"]["noise_seed"] = random.randint(1, 1000000000) #5555555 # For testing, use a fixed seed like 5555555
            
            # 4. Queue the Workflow
            response = self.queue_prompt(workflow)
            prompt_id = response['prompt_id']
            print(f"Comfy job queued! Prompt ID: {prompt_id}")

            # 5. Wait for completion
            self.track_progress(prompt_id)  # 10 minute timeout for video generation

            # 6. Download Result
            history = self.get_history(prompt_id)[prompt_id]
            saved_files = {}

            # Debug: Print history structure to diagnose issues
            print(f"DEBUG: History keys: {history.keys()}")
            if 'outputs' in history:
                print(f"DEBUG: Output node IDs: {list(history['outputs'].keys())}")
                for node_id, node_output in history['outputs'].items():
                    print(f"DEBUG: Node {node_id} output keys: {list(node_output.keys())}")
            else:
                print("DEBUG: No 'outputs' key in history!")

            output_dir = os.path.join(self.OUTPUT_DIR, folder)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            """# Save input image copy
            ts = int(time.time())
            input_ext = os.path.splitext(input_image_path)[1]
            start_filename = f"{ts}_start{input_ext}"
            try:
                with open(input_image_path, "rb") as f_in, open(os.path.join(output_dir, start_filename), "wb") as f_out:
                    f_out.write(f_in.read())
                print(f"Saved input image to {os.path.join(output_dir, start_filename)}")
                saved_files.append(os.path.join(output_dir, start_filename))

            except Exception as e:
                print(f"Error saving input image: {e}")"""
            
            ts = int(time.time())
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                
                # Check for videos (gifs/mp4) or images
                output_data = []
                if 'gifs' in node_output: output_data += node_output['gifs']
                if 'videos' in node_output: output_data += node_output['videos']
                if 'images' in node_output: output_data += node_output['images']

                for item in output_data:
                    print(f"Downloading {item['filename']}...")
                    file_data = self.download_file(item['filename'], item['subfolder'], item['type'])
                    
                    ext = os.path.splitext(item['filename'])[1]
                    if ext.lower() in ['.mp4', '.mov', '.avi', '.gif', '.webm']:
                        new_filename = f"{base_file_name}_{file_name_modifier}_video{ext}"
                        file_key = "video"
                    else:
                        new_filename = f"{base_file_name}_{file_name_modifier}_end{ext}"
                        file_key = "image_final"
                    
                    output_path = os.path.join(output_dir, new_filename)
                    with open(output_path, "wb") as f:
                        f.write(file_data)

                    print(f"Saved to {output_path}")
                    saved_files[file_key] = output_path

                # Also save the animation prompt, source image prompt, and source image review to a text file for reference
                #### THIS FILE NAME LOGIC NEED TO CHANGE WE MAY NOTE REVIEW EACH, SO WE ARE JUST COPYING REVIEW####
                prompt_path = os.path.join(output_dir, f"{base_file_name}_{file_name_modifier}_animation_prompt.txt")
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(prompt_text+"\n\n>>This line not in the prompt, <lora> tag used was: "+lora)
                source_img_path = os.path.join(output_dir, f"{base_file_name}_{file_name_modifier}_source_image_prompt.txt")
                with open(source_img_path, "w", encoding="utf-8") as f:
                    f.write(source_image_prompt)

                source_review_path = os.path.join(output_dir, f"{base_file_name}_{file_name_modifier}_source_image_review.txt")
                with open(source_review_path, "w", encoding="utf-8") as f:
                    f.write(source_image_review)

                saved_files["animation_prompt"] = prompt_path
                saved_files["source_image_prompt"] = source_img_path
                saved_files["source_image_review"] = source_review_path
                saved_files["image_start"] = input_image_path

                
            """#Free up models after inference to save VRAM
            print("SEDING REQUEST TO FREE MODELS FROM VRAM - waiting 5 seconds...")
            time.sleep(5)  # Wait a bit to ensure all processes are done
            requests.post(f'http://{self.COMFY_SERVER}/free', json={'unload_models': True})"""
            if clear_vram:
                self.aggressive_cleanup()  # Aggressively clean up VRAM
            return saved_files

        except Exception as e:
            print(f"An error occurred using ComfyCall: {e}")
            return {}


if __name__ == "__main__":
    # Example usage
    comfy = ComfyCall()
    comfy.aggressive_cleanup()
    # You can replace these with actual test values if needed
    #comfy.run(r"C:\Users\x\Documents\code\local_jarvis\xserver\autogen\jan20_diana\1768921264_5_end.png", "blowjob head motion. She rapidly moves her lips down the penis shaft, the penis goes all the way in to her mouth and throat until her lips are on his groin. Then pulls her head backwards  and the penis shaft almost exits her mouth. then goes back down towards his groin agian. repeats the motion quickly. Camera close-up POV",lora='blowjob')
    #comfy.edit_image(r"C:\Users\x\Documents\code\local_jarvis\xserver\autogen\shoots\1769052641_image.png", "The same woman sitting with legs spread, her arms leaning back behind her. direct eye contact with viewer. Keep hairstyle, clothing, and facial features identical. Real skin detail, detailed individual hair strands. remarkable eye details. low-level angle from her waist, medium-shot")