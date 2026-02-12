import cv2
import numpy as np
import sys
import os


def motion_opacity_maker(video_path, fps=16, timeframe=1, display_result=False, num_frames=3):
    """
    Create a motion blur composite image from the final frames of a video.
    Overlays multiple frames with varying opacity to create a shadow trail effect.
    
    Args:
        video_path: Path to input video file
        fps: Frames per second (default: 16)
        timeframe: Time window in seconds (default: 1)
        display_result: Whether to display result (default: False)
        num_frames: Number of frames to use for motion blur (default: 3)
    
    Returns:
        Composite image with motion blur effect showing frame transitions
    """
    # Calculate the time window in frames
    time_window_frames = int(fps * timeframe)
    
    # Ensure at least 1 frame is used
    num_frames = max(1, num_frames)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame positions evenly spaced across the time window
    last_frame_pos = total_frames - 1
    first_frame_pos = max(0, last_frame_pos - time_window_frames + 1)
    
    frame_positions = []
    for i in range(num_frames):
        if num_frames > 1:
            offset = int((time_window_frames - 1) * i / (num_frames - 1))
        else:
            offset = 0
        frame_pos = min(last_frame_pos, first_frame_pos + offset)
        frame_positions.append(frame_pos)
    
    # Read all frames
    frames = []
    for frame_pos in frame_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {frame_pos}")
            cap.release()
            return None
        frames.append(frame.astype(np.float32))
    
    cap.release()
    
    # Create opacity weights: older frames more transparent, newer frames more opaque
    # Adjust curve strength based on number of frames - fewer frames = gentler curve
    if num_frames <= 4:
        # Linear or gentle curve for fewer frames
        alphas = np.linspace(0.01, 1.0, num_frames)
    else:
        # More aggressive curve for more frames
        curve_strength = 1.2 + (num_frames - 4) * 0.1  # Scales from 1.2 to 1.6
        alphas = np.linspace(0.01, 1.0, num_frames) ** curve_strength
    
    # Create composite using maximum blending for motion trails
    # This preserves bright areas and creates ghost-like motion streaks
    composite = np.zeros_like(frames[0], dtype=np.float32)
    
    for frame, alpha in zip(frames, alphas):
        # Apply opacity to frame
        weighted_frame = frame * alpha
        # Use maximum to preserve bright moving objects across frames
        composite = np.maximum(composite, weighted_frame)
    
    # Convert back to uint8
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    
    print(f"Motion blur composite created using {num_frames} frames:")
    for i, (frame_pos, alpha) in enumerate(zip(frame_positions, alphas)):
        print(f"  Frame {frame_pos + 1}: opacity = {alpha:.2f}")
    print(f"Time window: {time_window_frames} frames ({timeframe}s at {fps} fps)")

    if display_result:
        cv2.imshow('Motion Opacity Composite - Press any key to close', composite)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return composite

def calculate_optical_flow_with_grid(video_path, grid_size=10, frame_offset=5, output_path=None, display_result=False, overlay_output=True, timeframe=1, down_scale=True):
    """
    Calculate optical flow from a video and visualize with arrows on a grid.
    
    Args:
        video_path: Path to input video file
        grid_size: Number of grid cells (grid_size x grid_size)
        frame_offset: Number of frames back from the end to use as the start frame (default: 5, which is about .5 second)
        output_path: Path to save output image (optional, auto-generated if None)
        display_result: Whether to display the result in a window (default: False)
        overlay_output: Whether to use motion opacity composite as base image (default: True)
        timeframe: Time window in seconds for motion opacity composite (default: 1)
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")
    
    # Calculate frame positions
    start_frame = max(0, total_frames - frame_offset - 1)
    end_frame = total_frames - 1
    
    # Read start frame (offset frames from the end)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame1 = cap.read()
    if not ret:
        print(f"Error: Could not read frame {start_frame}")
        cap.release()
        return
    
    # Read last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
    ret, frame2 = cap.read()
    if not ret:
        print("Error: Could not read last frame")
        cap.release()
        return
    
    # Get FPS before releasing the capture
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    cap.release()
    print(f"Comparing frame {start_frame + 1} to frame {end_frame + 1} (offset: {frame_offset} frames)")
    
    if down_scale:
        frame1 = down_scale_image(frame1)
        frame2 = down_scale_image(frame2)
    
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, 
        None, 
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    
    # Get frame dimensions
    h, w = frame1.shape[:2]
    
    # Calculate grid cell size
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    # Create output image
    if overlay_output:
        # Use motion opacity composite as base
        fps = video_fps if video_fps > 0 else 16
        output_img = motion_opacity_maker(video_path, fps=fps, timeframe=timeframe,display_result=display_result)
        
        if down_scale and output_img is not None:
            output_img = down_scale_image(output_img)
            
        if output_img is None:
            # Fallback to last frame if motion opacity maker fails
            output_img = frame2.copy()
    else:
        # Use last frame as base
        output_img = frame2.copy()
    
    # Calculate average flow for each grid cell and draw arrows
    for i in range(grid_size):
        for j in range(grid_size):
            # Define grid cell boundaries
            y_start = i * cell_h
            y_end = min((i + 1) * cell_h, h)
            x_start = j * cell_w
            x_end = min((j + 1) * cell_w, w)
            
            # Extract flow for this cell
            cell_flow = flow[y_start:y_end, x_start:x_end]
            
            # Calculate average flow in this cell
            avg_flow_x = np.mean(cell_flow[:, :, 0])
            avg_flow_y = np.mean(cell_flow[:, :, 1])
            
            # Calculate flow magnitude
            magnitude = np.sqrt(avg_flow_x**2 + avg_flow_y**2)
            
            # Calculate center point of the cell
            center_x = x_start + cell_w // 2
            center_y = y_start + cell_h // 2
            
            # Calculate end point of arrow (increased scale factor)
            scale_factor = 5  # Increased from 3 for better visibility
            end_x = int(center_x + avg_flow_x * scale_factor)
            end_y = int(center_y + avg_flow_y * scale_factor)
            
            # Draw arrow (green color)
            # Only draw arrows with noticeable motion
            if magnitude > 0.5:  # Threshold to avoid drawing tiny arrows
                cv2.arrowedLine(
                    output_img,
                    (center_x, center_y),
                    (end_x, end_y),
                    (0, 255, 0),  # Green color in BGR
                    thickness=2,
                    tipLength=0.3
                )
            else:
                # Draw a small circle for areas with minimal motion
                cv2.circle(output_img, (center_x, center_y), 3, (0, 255, 0), -1)
    
    # Calculate and print overall flow statistics
    flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    print(f"Flow magnitude - Min: {flow_magnitude.min():.2f}, Max: {flow_magnitude.max():.2f}, Mean: {flow_magnitude.mean():.2f}")
    
    # Display the result
    if display_result:
        cv2.imshow('Optical Flow - Press any key to close', output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Save the output image
    # Get the directory of the video file
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    if output_path is None:
        # Auto-generate output filename in video's directory
        output_path = os.path.join(video_dir, f"{video_name}_optical_flow.png")
    elif not os.path.isabs(output_path):
        # If output_path is just a filename, save it in the video's directory
        output_path = os.path.join(video_dir, output_path)
    # If output_path is an absolute path, use it as-is
    
    cv2.imwrite(output_path, output_img)
    print(f"Output saved to: {output_path}")
    
    return output_img, output_path

def down_scale_image(img, target_height=360, target_width=640):
    try:
        if img is None:
            return None
            
        # Use OpenCV for resizing
        resized_image = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            
        return resized_image
    except Exception as e:
        print(f"Error downscaling image: {e}")
        return img


if __name__ == "__main__":
    # Example usage
    video_path = r"C:\Users\jared\Documents\code\local_jarvis\xserver\z\video\ComfyUI_01843_.mp4"
    
    # Allow command line argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    # Optional: specify grid size, frame offset, and output path
    grid_size = 5  # 5x5 grid
    frame_offset = 4  # frames back from end
    
    if len(sys.argv) > 2:
        grid_size = int(sys.argv[2])
    
    if len(sys.argv) > 3:
        frame_offset = int(sys.argv[3])
    
    output_path = None
    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    
    print(f"Processing video: {video_path}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Frame offset: {frame_offset}")
    
    output_img, output_path = calculate_optical_flow_with_grid(video_path, grid_size, frame_offset, output_path)
