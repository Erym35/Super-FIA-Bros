import os
import sys
import warnings
import logging

# 1. Suppress Python Warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')

# 2. Suppress Tensorflow/CUDA Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# 3. Suppress Gym/Envs Logs (Aggressive)
gym_logger = logging.getLogger('gym')
gym_logger.setLevel(logging.ERROR)

# 4. Global Logging Level
logging.basicConfig(level=logging.ERROR) 

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import pickle
import cv2
import numpy as np

import argparse

def main():
    parser = argparse.ArgumentParser(description='Replay best actions')
    parser.add_argument('--level', type=str, default='1-1', help='Level to replay (e.g. 1-1, 1-2)')
    parser.add_argument('--file', type=str, default='best_actions.pkl', help='Name of the action file (default: best_actions.pkl)')
    args = parser.parse_args()

    # Paths
    local_dir = os.path.dirname(__file__)
    actions_path = os.path.join(local_dir, args.file)
    video_dir = os.path.join(local_dir, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    video_output_path = os.path.join(video_dir, f'mario_replay_{args.level}.mp4')
    
    # Check file
    if not os.path.exists(actions_path):
        print(f"Errore: {actions_path} non trovata. Run training until high fitness (>500) to generate it.")
        sys.exit(1)
        
    print(f"Caricando le azioni di {actions_path} per il livello {args.level}...")
    actions = pickle.load(open(actions_path, 'rb'))
    print(f"Caricate {len(actions)} azioni.")
    
    # Setup Env
    env_name = f'SuperMarioBros-{args.level}-v0'
    try:
        env = gym_super_mario_bros.make(env_name)
    except Exception as e:
        print(f"Errore caricamento livello {env_name}: {e}")
        sys.exit(1)

    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    try:
        env.seed(42)
        env.action_space.seed(42)
        if hasattr(env.unwrapped, 'seed'):
             try:
                 env.unwrapped.seed(42)
             except:
                 pass
    except Exception as e:
        print(f"Seeding warning: {e}")
        
    state = env.reset()
    
    # Video Recorder
    height, width, _ = env.observation_space.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Universal fallback
    video = cv2.VideoWriter(video_output_path, fourcc, 30.0, (width, height))
    
    info = {'x_pos': 0}
    
    print("Mario sta giocando...")
    
    for i, action in enumerate(actions):
        # Step
        state, reward, done, info = env.step(action)
        
        # Convert to BGR for OpenCV video writer
        frame = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)

        # --- RETRO HUD DRAWING ---
        current_x = info.get('x_pos', 0)
        hud_text = f"DISTANCE: {current_x}"
        
        # Font settings for Retro look (CUSTOMIZATION: Change font_scale to resize text)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4 # Smaller (was 0.5)
        thickness = 1
        text_color = (255, 255, 255) 
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(hud_text, font, font_scale, thickness)
        
        # Box Setup
        margin = 4 # Smaller margin (was 5)
        box_w = text_w + (margin * 4) + 10 # Less padding width
        box_h = text_h + (margin * 4)
        
        # Coordinates for Bottom Right placement (CUSTOMIZATION: Adjust subtraction to move box)
        h_frame, w_frame, _ = frame.shape
        x_box = w_frame - box_w - 5 # 5 pixels from Right edge
        y_box = h_frame - box_h - 4 # 4 pixels from Bottom edge (Lowered significantly)
        
        # 1. Create Transparent Background (Dark Blue)
        # Define ROI (Region of Interest)
        sub_img = frame[y_box:y_box+box_h, x_box:x_box+box_w]
        
        # Create dark blue rectangle (BGR: Dark Blue is ~ 139, 0, 0 in RGB -> 0, 0, 139 in BGR)
        # Let's go for a nice Deep Mario Blue/Black: (40, 0, 0)
        blue_rect = np.full(sub_img.shape, (80, 0, 0), dtype=np.uint8) 
        
        # Blend it (Alpha 0.6 for transparency)
        res = cv2.addWeighted(sub_img, 0.4, blue_rect, 0.6, 1.0)
        
        # Put back into frame
        frame[y_box:y_box+box_h, x_box:x_box+box_w] = res
        
        # 2. Draw Pixelated White Border around the box
        cv2.rectangle(frame, (x_box, y_box), (x_box+box_w, y_box+box_h), (255, 255, 255), 1, cv2.LINE_4)
        # Optional: Inner black border for contrast
        cv2.rectangle(frame, (x_box+2, y_box+2), (x_box+box_w-2, y_box+box_h-2), (0, 0, 0), 1, cv2.LINE_4)

        # 3. Draw Text
        # Text shadows (Black)
        cv2.putText(frame, hud_text, (x_box + margin + 10 + 2, y_box + box_h - margin - 5 + 2), 
                    font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_4)
        # Text Foreground (White)
        cv2.putText(frame, hud_text, (x_box + margin + 10, y_box + box_h - margin - 5), 
                    font, font_scale, text_color, thickness, cv2.LINE_4)
        # -------------------------
        
        # Write Frame
        video.write(frame)
        
        if done:
            if info['x_pos'] == 3161:
                 print("Yuppie! Mario ha completato il livello.")
                 
                 # HACK: Recursive Force "Undone"
                 # We need to find the layer that is holding the 'done' flag and reset it.
                 # Gym wrappers often shadow attributes, so we drill down.
                 current_env = env
                 while True:
                     if hasattr(current_env, 'done'):
                        current_env.done = False
                     if hasattr(current_env, 'env'):
                        current_env = current_env.env
                     else:
                        break
                 
                 # Victory Lap: Continued recording for ~10 seconds
                 for _ in range(500):
                     try:
                         # Send NO-OP (0) action to let the emulator run
                         state, _, _, info = env.step(0)
                         
                         # Force done=False again just in case the env resets it internally after a step
                         current_env = env
                         while True:
                             if hasattr(current_env, 'done'):
                                current_env.done = False
                             if hasattr(current_env, 'env'):
                                current_env = current_env.env
                             else:
                                break
                         
                         # STOP CONDITION:
                         # 1. If Stage changes (e.g. 1-1 to 1-2)
                         # 2. If x_pos resets (e.g. 3000 -> 40), it means a new level loaded.
                         if info.get('x_pos') == 3175:
                             break
                         
                         # --- DRAW HUD (Restored) ---
                         frame = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
                         
                         current_x = info.get('x_pos', 0)
                         hud_text = f"DISTANCE: {current_x}"
                         font = cv2.FONT_HERSHEY_SIMPLEX
                         font_scale = 0.4
                         thickness = 1
                         text_color = (255, 255, 255)
                         (text_w, text_h), baseline = cv2.getTextSize(hud_text, font, font_scale, thickness)
                         margin = 4
                         box_w = text_w + (margin * 4) + 10
                         box_h = text_h + (margin * 4)
                         h_frame, w_frame, _ = frame.shape
                         x_box = w_frame - box_w - 5
                         y_box = h_frame - box_h - 4
                         sub_img = frame[y_box:y_box+box_h, x_box:x_box+box_w]
                         blue_rect = np.full(sub_img.shape, (80, 0, 0), dtype=np.uint8) 
                         res = cv2.addWeighted(sub_img, 0.4, blue_rect, 0.6, 1.0)
                         frame[y_box:y_box+box_h, x_box:x_box+box_w] = res
                         cv2.rectangle(frame, (x_box, y_box), (x_box+box_w, y_box+box_h), (255, 255, 255), 1, cv2.LINE_4)
                         cv2.rectangle(frame, (x_box+2, y_box+2), (x_box+box_w-2, y_box+box_h-2), (0, 0, 0), 1, cv2.LINE_4)
                         cv2.putText(frame, hud_text, (x_box + margin + 10 + 2, y_box + box_h - margin - 5 + 2), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_4)
                         cv2.putText(frame, hud_text, (x_box + margin + 10, y_box + box_h - margin - 5), font, font_scale, text_color, thickness, cv2.LINE_4)
                         # --------------------------------------

                         video.write(frame)
                     except Exception as e:
                         print(f"Errore durante la generazione del video vittoria: {e}")
                         break
            else:
                 print("Mario è morto.")
            break
            
    print(f"Run terminata. Distanza finale: {info['x_pos']}")
    video.release()
    env.close()
    
    # Bugfix for Colab Display: 
    # Colab often looks for 'mario_replay.mp4' by default. We create a copy to ensure it shows the latest run.
    import shutil
    default_output_path = os.path.join(video_dir, 'mario_replay.mp4')
    try:
        shutil.copy(video_output_path, default_output_path)
        print(f"Synced video to default path: {default_output_path}")
    except Exception as e:
        print(f"Warning: Could not sync to default path: {e}")

    print(f"Video saved to {video_output_path}")

if __name__ == "__main__":
    main()
