#!/usr/bin/env python3
"""
Flask server for LUSTIFY image generation
Provides a REST API for text-to-image generation using ImageGenerator
"""

import os
from flask import Flask, request, jsonify
from lustify_gen import ImageGenerator
import time

app = Flask(__name__)


#when running in terminal use this to set the GPU
#$env:CUDA_VISIBLE_DEVICES="0"

# Global instance of ImageGenerator
generator = None


def get_generator(animated=False):
    """Get or initialize the global ImageGenerator instance"""
    global generator
    if generator is None:
        print(f"Initializing ImageGenerator...\n\nAnimated={animated}!\n\n")
        generator = ImageGenerator(use_mps=False, animated=animated)
    return generator


@app.route('/generate', methods=['POST'])
def generate_image():
    """
    Generate an image from a text prompt
    
    Expected JSON body:
    {
        "prompt": "your image description",
        "output_file": "filename.png",  # optional, defaults to timestamp
        "output_dir": "/path/to/output",  # optional
        "width": 1024,  # optional
        "height": 1024,  # optional
        "num_inference_steps": 30,  # optional
        "guidance_scale": 5.5  # optional
    }
    
    Returns:
    {
        "success": true/false,
        "output_path": "/path/to/generated/image.png",
        "error": "error message if failed"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({
                "success": False,
                "error": "Missing required field: prompt"
            }), 400
        
        prompt = data['prompt']
        output_file = data.get('output_file', f"{time.time()}_generated.png")
        
        # Optional parameters
        kwargs = {}
        if 'output_dir' in data:
            kwargs['output_dir'] = data['output_dir']
        if 'width' in data:
            kwargs['width'] = data['width']
        if 'height' in data:
            kwargs['height'] = data['height']
        if 'num_inference_steps' in data:
            kwargs['num_inference_steps'] = data['num_inference_steps']
        if 'guidance_scale' in data:
            kwargs['guidance_scale'] = data['guidance_scale']
        if 'negative_prompt' in data:
            kwargs['negative_prompt'] = data['negative_prompt']
        if 'shoot_folder' in data:
            kwargs['shoot_folder'] = data['shoot_folder']
        
        gen = get_generator()
        output_path = gen.text_to_image(prompt, output_file=output_file, **kwargs)
        
        if output_path:
            return jsonify({
                "success": True,
                "output_path": output_path
            })
        else:
            return jsonify({
                "success": False,
                "error": "Image generation failed"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "generator_loaded": generator is not None
    })


@app.route('/unload', methods=['POST'])
def unload_model():
    """Unload the model to free up memory"""
    global generator
    try:
        if generator:
            generator._unload_model()
            generator = None
        return jsonify({
            "success": True,
            "message": "Model unloaded successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    
    #when running in terminal use this to set the GPU
    #$env:CUDA_VISIBLE_DEVICES="0"

    #set to GPU 1 by default
    #import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print("CUDA_VISIBLE_DEVICES set to 1")
    # Pre-load the generator on startup
    print("Starting LUSTIFY Image Generation Server...")
    get_generator(animated=False)
    
    # Run the Flask server
    app.run(host='0.0.0.0', port=8052, debug=False)
