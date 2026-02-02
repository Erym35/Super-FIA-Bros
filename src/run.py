import neat
import os
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
import pickle
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import visualize
import cv2
import neat.genome

ACTIONS = SIMPLE_MOVEMENT

def main(config_file, file, level="1-1"):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    genome = pickle.load(open(file, 'rb'))
    env = gym_super_mario_bros.make('SuperMarioBros-'+level+'-v0')
    env = JoypadSpace(env, ACTIONS)
    
    # Setup Video Writer
    height, width, _ = env.observation_space.shape
    video_dir = os.path.join(os.path.dirname(__file__), 'videos')
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, 'mario_run.mp4')

    
    # Try avc1 codec 
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    video = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
    
    if not video.isOpened():
        print(f"Error: Could not open video writer for {video_path}")
        print("Trying fallback codec: mp4v")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        if not video.isOpened():
            print("Error: Could not open video writer with mp4v either.")

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    visualize.draw_net(config, genome, True, filename="mario_net")
    info = {'x_pos': 0}
    
    print("Recording video to {}...".format(video_path))
    
    try:
        # Robust seeding
        try:
            env.seed(42)
            env.action_space.seed(42)
        except Exception as e:
            print(f"Warning: Standard seeding failed: {e}")
            
        # Try to seed the unwrapped nes_py environment directly if possible
        if hasattr(env.unwrapped, 'seed'):
             try:
                 env.unwrapped.seed(42)
             except:
                 pass
                 
        state = env.reset()

        # Write first frame
        if video.isOpened():
            video.write(cv2.cvtColor(state, cv2.COLOR_RGB2BGR))
        
        done = False
        i = 0
        old = 40
        while not done:
            # Prepare state for NEAT (Grayscale, Resized)
            state_input = cv2.resize(state, (13, 16))
            state_input = cv2.cvtColor(state_input, cv2.COLOR_RGB2GRAY)
            state_input = state_input.flatten()
            
            output = net.activate(state_input)
            ind = output.index(max(output))
            
            # Step environment
            state, reward, done, info = env.step(ind)
            
            # Write frame to video (Convert RGB to BGR for OpenCV)
            if video.isOpened():
                video.write(cv2.cvtColor(state, cv2.COLOR_RGB2BGR))
            
            i += 1
            if i % 50 == 0:
                print(f"Step {i}: Action={ind}, X_Pos={info['x_pos']}, Reward={reward}")
                if old == info['x_pos']:
                    print(f"STUCK at {info['x_pos']}! Ending run.")
                    break
                else:
                    old = info['x_pos']
                    
        print("Distance: {}".format(info['x_pos']))
        
    except KeyboardInterrupt:
        print("Interrupted!")
    finally:
        video.release()
        env.close()
        print("Video saved successfully.")


if __name__ == "__main__":
    CONFIG = 'config' # Added default config for standalone run
    main(CONFIG, "winner.pkl")
