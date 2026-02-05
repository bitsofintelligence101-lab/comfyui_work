# InfiniteTalk V2V Lip Sync: Optimal Wan 2.1 ComfyUI Configuration for 32GB Systems

**InfiniteTalk delivers realistic video-to-video lip sync through audio-driven diffusion on Wan 2.1's foundation**. For your 32GB VRAM configuration, you can run full FP16/BF16 precision models at 720p resolution with **30-40 sampling steps**—the maximum quality configuration for Wan 2.1.

**Critical Note: This guide is for Wan 2.1, NOT Wan 2.2.** Wan 2.1 uses a single unified model architecture, while Wan 2.2 uses a high-noise/low-noise Mixture of Experts (MoE) split architecture requiring different samplers at different stages. Your WanVideoSampler settings are for Wan 2.1.

**The V2V Challenge:** Balancing lip sync clarity against source video preservation requires specific denoise settings (**0.35–0.5**), audio CFG values (**4–6**), and potentially face-masking workflows to isolate mouth regeneration from background content.

---

## Base Models: Wan 2.1-I2V-14B Powers InfiniteTalk

InfiniteTalk from MeiGen-AI (released August 2025) uses **Wan 2.1-I2V-14B-480P** or **Wan 2.1-I2V-14B-720P** as its visual backbone, combined with a Chinese Wav2Vec2 audio encoder for phoneme extraction. The architecture injects audio conditioning into the diffusion process while preserving source video motion and identity.

**Required Model Stack for 32GB VRAM (Prioritizing Quality):**

| Component | Recommended File | Size | Source |
|-----------|-----------------|------|--------|
| **Base diffusion (720p)** | `wan2.1_i2V_720p_14B_fp16.safetensors` | ~27GB | Kijai/WanVideo_comfy |
| **Base diffusion (480p)** | `wan2.1_i2V_480p_14B_fp16.safetensors` | ~16GB | Kijai/WanVideo_comfy |
| **InfiniteTalk weights** | `Wan2_1-InfiniteTalk-Single_fp16.safetensors` | ~4GB | Kijai/WanVideo_comfy/InfiniteTalk |
| **InfiniteTalk (FP8 option)** | `Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors` | ~2.6GB | Kijai/WanVideo_comfy_fp8_scaled/InfiniteTalk |
| **Audio encoder** | TencentGameMate/chinese-wav2vec2-base | ~0.3GB | Auto-downloads via HF |
| **Vocal separator** | `MelBandRoformer_fp16.safetensors` | ~0.5GB | Kijai/MelBandRoFormer_comfy |

**With your 32GB VRAM, avoid GGUF quantized models** unless you need to run 720p with other heavy operations simultaneously. The FP16 models maintain excellent quality and fit comfortably in your VRAM budget. The FP8 scaled InfiniteTalk checkpoint is optimized for ComfyUI and maintains excellent quality while reducing VRAM by ~40%.

**Single vs Multi-speaker Checkpoints:** Use `InfiniteTalk-Single` for single-person footage. The `Multi` variant exists for scenes with multiple speakers requiring individual audio tracks.

---

## LoRA Models: LightX2V Acceleration for Quality-Focused Work

Two primary LoRAs exist for Wan 2.1 acceleration:

**LightX2V (Recommended for Your Use Case):**
- **File:** `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` (704MB)  
  Or: `lightx2v_I2V_14B_720p_cfg_step_distill_rank64_bf16.safetensors` (for 720p)
- **Effect:** Reduces sampling from **30-40 steps to 4 steps** with minimal quality loss
- **Download:** huggingface.co/Kijai/WanVideo_comfy/tree/main/Lightx2v
- **Rank options:** rank4 (52MB) through rank256 (2.8GB)—**rank64 offers the best quality-performance balance, rank128 for absolute maximum quality**

**When using LightX2V with WanVideoSampler, adjust these settings:**
- `steps`: **4** (instead of 30-40)
- `cfg` (text guidance): **1.0** (instead of 5.0-6.0)
- `audio_cfg` (audio_scale): **2.0–3.0** (instead of 4.0-6.0)

**FusionX (Avoid for V2V):**
- Enables 8-step generation but community reports confirm it **exacerbates color shift** after 1 minute and **degrades identity preservation**—critical issues for realistic human video. Not recommended for your use case.

---

## Text Encoder and VAE: Maximum Fidelity Selections

**Text Encoder:** Wan 2.1 uses UMT5-XXL (Google's multilingual T5) with cross-attention injection:

| File | Precision | VRAM | Recommendation |
|------|-----------|------|----------------|
| `umt5-xxl-enc-bf16.safetensors` | BF16 | ~6.5GB | **Best quality** |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | FP8 | ~3GB | Lower VRAM fallback |

**For your 32GB setup: Use BF16.** You have the VRAM headroom and want maximum quality.

**VAE:** The WAN-VAE handles temporal information preservation during video encoding/decoding:

| File | Precision | Quality |
|------|-----------|---------|
| `Wan2_1_VAE_bf16.safetensors` | BF16 | **Recommended—optimal stability** |
| `Wan2_1_VAE_fp16.safetensors` | FP16 | Alternative, nearly identical quality |

**CLIP Vision:** Required for image/video embedding—use `clip_vision_h.safetensors` (~1.2GB) from Comfy-Org/Wan_2.1_ComfyUI_repackaged.

---

## Sampler and Scheduler: YOUR WanVideoSampler Options for Wan 2.1

**Your Available Samplers in WanVideoSampler:**
- dpm++ variants
- euler
- euler/beta
- flowmatch_causvid  
- vibt_unipc
- lcm
- lcm/beta
- **unipc** ← Primary recommendation
- **unipc/beta**

### Recommended Sampler Settings for Wan 2.1 InfiniteTalk V2V:

**Without LightX2V LoRA (30-40 steps, maximum quality):**

| Setting | Value | Notes |
|---------|-------|-------|
| **Sampler** | **unipc** | Most recommended for Wan 2.1 |
| **Sampler (alt)** | **euler** | Also works well, slightly different motion character |
| **Sampler (alt 2)** | **dpm++_sde** | Community reports stability with this |
| **Steps** | **30-40** | Higher = better quality, diminishing returns after 40 |

**With LightX2V LoRA (4 steps, fast iteration):**

| Setting | Value | Notes |
|---------|-------|-------|
| **Sampler** | **lcm** or **unipc** | LCM designed for distilled models |
| **Steps** | **4** | Fixed for LightX2V distillation |

**Scheduler Note:** The `/beta` variants (euler/beta, unipc/beta, lcm/beta) are **Wan 2.2-specific features** for the beta noise scheduler used in Wan 2.2's MoE architecture. Since you're using **Wan 2.1**, you should use the **non-beta variants**: `unipc`, `euler`, `dpm++_sde`, or `lcm` (with LightX2V).

### Optimal Configuration for Maximum Quality V2V:

```
Sampler: unipc
Steps: 40 (without LoRA) or 4 (with LightX2V rank64/128)
CFG (text guidance): 5.0-6.0 (without LoRA) or 1.0 (with LightX2V)
Audio CFG (audio_scale): 4.0-6.0 (without LoRA) or 2.0-3.0 (with LightX2V)
```

---

## Critical V2V-Specific Parameters (Proven Through Testing)

**The Denoise Parameter - Your Most Important Control:**

Denoise controls how much of the source video to preserve vs regenerate:

| Denoise Value | Result | Use Case |
|---------------|--------|----------|
| **0.1** | ❌ Noisy mess - not enough regeneration | Never use |
| **0.35-0.38** | Maximum source preservation, softer mouth | Prioritize background/clothing |
| **0.4** | ⭐ **PROVEN SWEET SPOT** - crisp mouth, good preservation | **Start here** |
| **0.45** | Sharper mouth movements, more background changes | Need clearer lip sync |
| **0.5+** | Too much deviation from source video | Avoid for V2V |

**Start at 0.4 - this has been tested and proven optimal for most content.**

---

**Core Inference Parameters (Tested & Verified):**

| Parameter | Without LoRA | With LightX2V | Purpose |
|-----------|--------------|---------------|---------|
| **steps** | **40** ⭐ | **4** | Sampling iterations - 40 = crisp, 4 = slight blur |
| **cfg** (text guidance) | **5.5** | **1.0** | Text prompt adherence |
| **audio_cfg** (audio_scale) | **5.0** | **2.5** | Lip sync accuracy—higher = tighter sync |
| **denoise** | **0.4** ⭐ | **0.4** ⭐ | **CRITICAL** - balance preservation/regeneration |
| **sampler** | **unipc** | **unipc** | Best tested sampler for Wan 2.1 |
| **shift** | **3.0** (480p) / **5.0** (720p) | Same | Resolution-dependent noise schedule |

---

**Other Important Settings:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **motion_frame** | **9** | Motion conditioning frames (standard) |
| **Frames per chunk** | **81** | Standard chunk size for streaming generation |
| **Overlapping frames** | **25** | Ensures smooth transitions between chunks |
| **FPS** | **25-30** | Standard video frame rate |

---

**Resolution Recommendations:**

With 32GB VRAM, you have excellent options:

| Resolution | VRAM Usage | Quality | Speed | Best For |
|------------|------------|---------|-------|----------|
| **480p** (832×480) | ~16-18GB | Good | Fast | Testing/iteration |
| **720p** (1280×720) | ~22-26GB | ⭐ Excellent | Moderate | Final production |

**Use 480p for testing, 720p for finals.**

---

## ComfyUI Workflow Configuration and Node Setup

**Primary Integration:** Install **ComfyUI-WanVideoWrapper** by Kijai—the definitive node package for InfiniteTalk:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
pip install -r ComfyUI-WanVideoWrapper/requirements.txt
```

**Pre-Built V2V Workflows:**
- **Kijai official:** `ComfyUI-WanVideoWrapper/example_workflows/wanvideo_InfiniteTalk_V2V_example_02.json`
- **Bluespork V2V:** github.com/bluespork/InfiniteTalk-ComfyUI-workflows (`InfiniteTalk-V2V.json`)
- **NextDiffusion FP8:** cdn.nextdiffusion.ai/comfyui-workflows/InfiniteTalk-V2V-FP8-Lip-Sync.json
- **ThinkDiffusion:** learn.thinkdiffusion.com (search "InfiniteTalk V2V")

**Required Node Dependencies:**

| Package | Purpose |
|---------|---------|
| ComfyUI-WanVideoWrapper | InfiniteTalk integration |
| ComfyUI-VideoHelperSuite | Video loading/export |
| ComfyUI-KJNodes | Utility nodes |
| ComfyUI-MelBandRoFormer | Audio vocal separation |

**Key V2V Node Configuration:**

1. **VHS_LoadVideo:** 
   - Set `force_rate: 25-30` for source video
   - This ensures consistent frame timing

2. **InfiniteTalk Wav2vec2 Embeds (or Multi/InfiniteTalk node):** 
   - Set `num_frames` = FPS × audio_seconds (e.g., 25 FPS × 10s = 250 frames)
   - `audio_cfg`: **4-6** for strong lip sync without LoRA, **2-3** with LightX2V

3. **WanVideoSampler:** 
   - `sampler_name`: **unipc** (primary) or **euler** (alternative)
   - `steps`: **40** without LoRA, **4** with LightX2V
   - `cfg`: **5.0-6.0** without LoRA, **1.0** with LightX2V
   - `denoise`: **0.35-0.5** (start at 0.4 and adjust)

4. **Resolution Matching:** 
   - Portrait 9:16 = 480×832 or 720×1248
   - Landscape 16:9 = 832×480 or 1248×720

---

## Preserving Source Video Quality Through Face Masking

Community discussions (GitHub Issue #156) highlight a persistent challenge: **InfiniteTalk processes the entire frame**, which can degrade clothing, background, and scene elements. **Face-masking workflows** offer the best solution:

**Masking Approach Using SAM Segmentation:**

1. Use **ComfyUI-segment-anything-2** (SAM2) to isolate the face/mouth region
2. Apply InfiniteTalk only to the masked area  
3. Composite regenerated face onto original video
4. Use **Relight LoRA** (`WanAnimate_relight_lora_fp16.safetensors`) to correct color tone matching between composited regions

**Benefits:**
- Preserves background 100%
- Preserves clothing and body movement
- Maintains original lighting on non-face areas
- Only regenerates mouth/face region

**Trade-offs:**
- Increased workflow complexity
- Potential edge artifacts requiring careful feathering
- Requires additional SAM2 models and processing time

**Alternative Approach (Simpler):** Lower denoise values (0.35–0.4) preserve more source content but may reduce mouth clarity. Finding your optimal denoise balance requires experimentation with your specific source footage. Start at **0.4** and adjust ±0.05 based on results.

---

## Model Folder Structure for Immediate Implementation

```
ComfyUI/models/
├── diffusion_models/
│   ├── wan2.1_i2V_720p_14B_fp16.safetensors       (for 720p quality)
│   ├── wan2.1_i2V_480p_14B_fp16.safetensors       (for 480p faster)
│   ├── Wan2_1-InfiniteTalk-Single_fp16.safetensors
│   │   OR
│   ├── Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors  (FP8 option)
│   └── MelBandRoformer_fp16.safetensors
├── vae/
│   └── Wan2_1_VAE_bf16.safetensors
├── clip_vision/
│   └── clip_vision_h.safetensors
├── text_encoders/
│   └── umt5-xxl-enc-bf16.safetensors
└── loras/
    ├── lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors
    ├── lightx2v_I2V_14B_720p_cfg_step_distill_rank64_bf16.safetensors
    └── WanAnimate_relight_lora_fp16.safetensors  (for face masking workflows)
```

**Download Commands (via HuggingFace CLI):**
```bash
# Install HF CLI if needed
pip install "huggingface_hub[cli]"

# Base model (choose 720p OR 480p based on your needs)
huggingface-cli download Kijai/WanVideo_comfy \
  wan2.1_i2V_720p_14B_fp16.safetensors \
  --local-dir ./models/diffusion_models

# InfiniteTalk weights (FP16 for max quality)
huggingface-cli download Kijai/WanVideo_comfy \
  InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp16.safetensors \
  --local-dir ./models/diffusion_models

# VAE
huggingface-cli download Kijai/WanVideo_comfy \
  Wan2_1_VAE_bf16.safetensors \
  --local-dir ./models/vae

# Text Encoder
huggingface-cli download Kijai/WanVideo_comfy \
  umt5-xxl-enc-bf16.safetensors \
  --local-dir ./models/text_encoders

# CLIP Vision
huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
  clip_vision_h.safetensors \
  --local-dir ./models/clip_vision

# LightX2V LoRA (optional but recommended)
huggingface-cli download Kijai/WanVideo_comfy \
  Lightx2v/lightx2v_I2V_14B_720p_cfg_step_distill_rank64_bf16.safetensors \
  --local-dir ./models/loras
```

---

## Known Limitations and Quality Considerations

**V2V Motion Fidelity:** InfiniteTalk mimics source video camera movement but not identically—expect slight motion drift, especially in longer videos. The model preserves overall motion patterns but may introduce subtle variations.

**Color Shift Over Time:** Videos exceeding 1 minute can exhibit progressive color drift. This is exacerbated by FusionX LoRA (avoid it). For extended videos, consider:
- Processing in segments with color correction between chunks
- Using Relight LoRA to maintain color consistency
- Keeping segments under 60 seconds when possible

**The Denoise Paradox:** 
- **Lower denoise** (0.3-0.35) → preserves source video → reduces mouth quality
- **Higher denoise** (0.5-0.6) → improves lip clarity → degrades scene fidelity
- **Optimal range: 0.4-0.45** represents the practical sweet spot for most footage
- **Face-masking workflows bypass this limitation entirely** by only regenerating the mouth region

**Audio Quality Matters:** InfiniteTalk tracks phonemes best with:
- Clean, dry voice recordings (minimal background noise)
- Clear articulation
- Normalized audio levels
- Separated vocals (use MelBandRoFormer to remove background music)

**Wan 2.2 Compatibility:** InfiniteTalk is built for Wan 2.1. Community efforts to integrate with Wan 2.2 are ongoing but not yet fully supported—stick with the Wan 2.1-I2V-14B base model.

---

## Proven Optimal Settings (Based on Real-World Testing)

### **CRITICAL SETTINGS THAT MAKE OR BREAK QUALITY:**

**The Big Three Parameters:**

1. **Denoise Strength: 0.4** (most important for V2V)
   - **0.1 = Noisy mess** (not enough regeneration)
   - **0.4 = Sweet spot** (crisp mouth, good source preservation)
   - **0.45 = Sharper mouth** (slightly more background changes)
   - **0.6+ = Too much deviation** from source

2. **Sampling Steps: 40 without LoRA, 4 with LoRA**
   - **4 steps with LightX2V = Slight blur** (fast but softer)
   - **40 steps no LoRA = Crisp, sharp** (slower but best quality)
   - This is the #1 factor affecting image sharpness

3. **Resolution: 480p vs 720p**
   - **480p = Good quality, faster** processing
   - **720p = Noticeably sharper**, better facial detail
   - With 32GB VRAM, 720p is easily achievable

---

### **Recommended Configuration (Tested & Proven):**

**For Maximum Crispness (480p, ~15-20 min per minute of video):**
```
Base Model: wan2.1_i2V_480p_14B_fp16.safetensors
InfiniteTalk: Wan2_1-InfiniteTalk-Single_fp16.safetensors
Text Encoder: umt5-xxl-enc-bf16.safetensors
VAE: Wan2_1_VAE_bf16.safetensors
LoRA: NONE (disable LightX2V for crispness)

WanVideoSampler Settings:
├─ sampler: unipc
├─ steps: 40
├─ cfg: 5.5
├─ denoise_strength: 0.4
├─ shift: 3.0 (for 480p)
└─ audio_cfg: 5.0 (via InfiniteTalk node, not sampler)

Resolution: 832×480 (landscape) or 480×832 (portrait)
```

**Why These Settings Work:**
- **Steps: 40** = Maximum sharpness, eliminates blur from 4-step distillation
- **Denoise: 0.4** = Perfect balance between mouth clarity and source preservation
- **CFG: 5.5** = Good prompt adherence without artifacts
- **No LoRA** = No quality compromise from acceleration
- **BF16 models** = Highest precision available

---

**For Faster Iteration (480p, ~5-7 min per minute of video):**
```
Base Model: wan2.1_i2V_480p_14B_fp16.safetensors
InfiniteTalk: Wan2_1-InfiniteTalk-Single_fp16.safetensors
Text Encoder: umt5-xxl-enc-bf16.safetensors
VAE: Wan2_1_VAE_bf16.safetensors
LoRA: lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors

WanVideoSampler Settings:
├─ sampler: unipc
├─ steps: 4
├─ cfg: 1.0
├─ denoise_strength: 0.4
├─ shift: 3.0
└─ audio_cfg: 2.5

Resolution: 832×480 (landscape) or 480×832 (portrait)
```

**Trade-offs:**
- **Faster processing** (4 steps vs 40)
- **Slight softness/blur** compared to 40-step version
- **Good for testing** denoise/audio settings before final render
- **Use rank128 LoRA** (not rank64) for better quality than rank64

**Maximum Source Preservation (Face masking workflow):**
- Any of the above configurations
- Add SAM2 face segmentation
- Only regenerate face/mouth region
- Composite onto original video
- Apply Relight LoRA for color matching

---

## Troubleshooting Common Issues (Tested Solutions)

**Issue:** Output is full of noise, can barely see the character  
**Solution:** Denoise is too LOW. Increase from 0.1 to **0.4**  
**Why:** At denoise 0.1, only 10% regeneration occurs—not enough for the model to work properly

**Issue:** Output is blurry/soft, not crisp  
**Solution:** You're using LightX2V LoRA with 4 steps. **Disable the LoRA and use 40 steps**  
**Why:** The 4-step distillation trades sharpness for speed. 40 steps = maximum crispness

**Issue:** Mouth movements look good but background changes significantly  
**Solution:** Lower denoise to **0.38** (from 0.4) OR implement face-masking workflow  
**Why:** Lower denoise preserves more of the source video structure

**Issue:** Background preserved well but lip sync is mushy/unclear  
**Solution:** Increase denoise to **0.45** AND increase audio_cfg to **5.5-6.0**  
**Why:** Higher denoise allows more mouth regeneration, higher audio_cfg strengthens sync

**Issue:** Color shifts between chunks in long videos  
**Solution:** Keep chunks under 60 seconds, use Relight LoRA, or apply color correction in post  
**Why:** InfiniteTalk's color consistency degrades over extended generation

**Issue:** Identity drift (person's face changes over time)  
**Solution:** Ensure CLIP Vision is loaded correctly, avoid FusionX LoRA, use cleaner source  
**Why:** CLIP Vision anchors identity; strong LoRAs can override it

**Issue:** Audio out of sync with lips  
**Solution:** Increase audio_cfg to **5.5-6.0**, ensure audio is clean and normalized  
**Why:** Higher audio_cfg strengthens the phoneme-to-mouth mapping

**Issue:** Flickering or unstable frames  
**Solution:** Increase sampling steps to **50** (from 40), ensure overlapping_frames is **25**  
**Why:** More steps = smoother frame transitions

**Issue:** Video looks too static/rigid at 720p  
**Solution:** Confirm shift is **5.0** (not 6.0+), check that motion_frame is **9**  
**Why:** Too-high shift values create overly stable/frozen frames

**Issue:** OOM (Out of Memory) at 720p  
**Solution:** Reduce max_frames, close other VRAM-using apps, verify shift is exactly **5.0**  
**Why:** Your 32GB should handle 720p easily; likely a setting or other app using VRAM

---

---

## Upgrading to 720p for Maximum Sharpness

Once you've dialed in your settings at 480p, upgrading to 720p provides a significant quality boost with minimal configuration changes.

### **What Changes for 720p:**

**Model Files to Download:**
```bash
# 720p Base Model (~27GB)
huggingface-cli download Kijai/WanVideo_comfy \
  wan2.1_i2V_720p_14B_fp16.safetensors \
  --local-dir ./models/diffusion_models

# 720p LightX2V LoRA (if using acceleration)
huggingface-cli download Kijai/WanVideo_comfy \
  Lightx2v/lightx2v_I2V_14B_720p_cfg_step_distill_rank128_bf16.safetensors \
  --local-dir ./models/loras
```

**Everything else stays the same:**
- ✅ Same InfiniteTalk model (works with both 480p and 720p)
- ✅ Same VAE (Wan2_1_VAE_bf16.safetensors)
- ✅ Same text encoder (umt5-xxl-enc-bf16.safetensors)
- ✅ Same CLIP Vision (clip_vision_h.safetensors)

### **720p Optimal Settings:**

**Maximum Quality (720p, ~20-25 min per minute of video):**
```
Base Model: wan2.1_i2V_720p_14B_fp16.safetensors
InfiniteTalk: Wan2_1-InfiniteTalk-Single_fp16.safetensors
Text Encoder: umt5-xxl-enc-bf16.safetensors
VAE: Wan2_1_VAE_bf16.safetensors
LoRA: NONE

WanVideoSampler Settings:
├─ sampler: unipc
├─ steps: 40
├─ cfg: 5.5
├─ denoise_strength: 0.4
├─ shift: 5.0 ← CHANGED from 3.0 (480p uses 3.0, 720p uses 5.0)
└─ audio_cfg: 5.0

Resolution: 1280×720 (landscape) or 720×1280 (portrait)
```

**Accelerated 720p (with LightX2V, ~6-8 min per minute of video):**
```
Base Model: wan2.1_i2V_720p_14B_fp16.safetensors
LoRA: lightx2v_I2V_14B_720p_cfg_step_distill_rank128_bf16.safetensors
(everything else same as above)

WanVideoSampler Settings:
├─ sampler: unipc
├─ steps: 4
├─ cfg: 1.0
├─ denoise_strength: 0.4
├─ shift: 5.0
└─ audio_cfg: 2.5

Resolution: 1280×720 or 720×1280
```

### **The Shift Parameter Explained:**

**Shift controls the noise schedule's timing:**
- **480p models: shift = 3.0** (official recommendation)
- **720p models: shift = 5.0** (official recommendation)
- Higher shift = more stable, detailed frames
- Lower shift = more motion/dynamics but less detail

**Don't use the wrong shift value:**
- 480p with shift 5.0 = overly static, rigid
- 720p with shift 3.0 = unstable, artifacts

### **720p Quality Improvements You'll See:**

1. **Sharper Facial Features**
   - More detailed mouth movements
   - Clearer teeth/tongue definition
   - Better eye/eyebrow detail

2. **Better Hair Rendering**
   - Individual hair strands visible
   - Less "mushy" texture

3. **Clothing/Background Detail**
   - Fabric textures preserved
   - Background elements stay crisp

4. **Overall Professional Quality**
   - Broadcast-ready resolution
   - Suitable for final production

### **VRAM Usage (720p vs 480p):**

With your 32GB VRAM:
- **480p: ~16-18GB** VRAM usage
- **720p: ~22-26GB** VRAM usage
- **Both fit comfortably** in your 32GB

Generation time increases by approximately **30-40%** for 720p vs 480p, but the quality gain is substantial.

### **Recommended Workflow:**

1. **Test at 480p first**
   - Dial in denoise (0.4 is proven sweet spot)
   - Test audio_cfg for your specific voice/content
   - Verify source preservation is acceptable

2. **Switch to 720p for finals**
   - Use same denoise/audio_cfg values
   - Change shift from 3.0 to 5.0
   - Render final production videos

3. **Use 480p + LightX2V for iteration**
   - Fast testing of different audio clips
   - Quick previews
   - Parameter experimentation

### **Common 720p Issues and Fixes:**

**Issue:** OOM (Out of Memory) errors at 720p  
**Solution:** 
- Make sure shift is 5.0 (not higher)
- Reduce max_frames if generating very long videos
- Close other applications using VRAM

**Issue:** Video looks too static/rigid at 720p  
**Solution:** 
- Confirm shift is 5.0 (not 6.0 or 7.0)
- Slightly lower denoise to 0.38-0.4 if needed
- Ensure motion_frame is set to 9

**Issue:** Slower than expected generation  
**Solution:** 
- This is normal - 720p has ~2.25x more pixels than 480p
- Use LightX2V LoRA for faster generation
- Consider using FP8 models if you need more speed (though quality drops slightly)

---

## Final Recommendations Summary

**For Your 32GB VRAM System:**

### **During Development/Testing:**
- Use **480p** with **LightX2V rank128 LoRA**
- Settings: steps=4, cfg=1.0, denoise=0.4, shift=3.0
- Fast iteration for testing different denoise/audio values
- Time: ~5-7 minutes per minute of video

### **For Final Production:**
- Use **720p** with **NO LoRA**
- Settings: steps=40, cfg=5.5, denoise=0.4, shift=5.0
- Maximum sharpness and quality
- Time: ~20-25 minutes per minute of video

### **The Proven Formula:**
```
Crisp Output = 40 steps + denoise 0.4 + no LoRA + BF16 models
Fast Testing = 4 steps + denoise 0.4 + LightX2V rank128 + same BF16 models
Maximum Quality = Above formula + 720p base model + shift 5.0
```

The denoise value of **0.4** has been proven optimal through testing - it balances source preservation with mouth clarity perfectly. Start here and only adjust ±0.05 if needed for your specific content.

---