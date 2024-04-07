from pettingzoo.mpe import simple_spread_v3
import numpy as np
env = simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False, render_mode="human")
#env = simple_spread_v3.env(render_mode="human")
def action_steps(file_path):
    import numpy as np
    import re
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    avg_reward = []
    
    with open(file_path, 'rb') as f:
        event_acc = EventAccumulator(f.name)
        # Load the log
        event_acc.Reload()
        tags = event_acc.Tags()
        print(tags)

        # ==================== Should Change to 'episode_acts/text_summary' =================== #
        tag = 'episode_acts/text_summary'
        acts_str = event_acc.Tensors(tag)
        # Extract the last episode's actions
        data_str = acts_str[-1].tensor_proto.string_val[0]
        # Decode to utf-8
        tensor_string = data_str.decode('utf-8')
        # Load the numbers from the string to the list
        numbers = re.findall(r'\d+\.\d+', tensor_string)
        float_numbers = [float(num) for num in numbers]
        int_numbers = [int(num) for num in float_numbers]
        #print(int_numbers)
        # Store these actions into a array, size = 25*3*5 (episode_length * num_agent * num_actions)
        acts = np.array(int_numbers)
        acts_array = np.reshape(acts, newshape=(3,25,5))
        print(acts_array)
        
        
    return acts_array
# ===================================== Change the file path ============================= #
#file_path = 'D:/MAAI/off-policy/offpolicy/scripts/results/MPE/simple_spread/qmix/debug/run28/logs/events.out.tfevents.1712228989.DESKTOP-U9OC6U6'
#file_path = 'D:/MAAI/off-policy/offpolicy/scripts/results/MPE/simple_spread/qmix/debug/run33/logs/events.out.tfevents.1712397982.DESKTOP-U9OC6U6' # 33: our test environment; 34: example env
file_path = 'D:/MAAI/off-policy/offpolicy/scripts/results/MPE/simple_spread/qmix/debug/run39/logs/events.out.tfevents.1712437990.DESKTOP-U9OC6U6'
acts_array = action_steps(file_path)

env.reset()

i = 0
step = 0

for agent in env.agent_iter():
    
    observation, reward,termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
        env.step(action)
        print(action)
        continue
        
    if i%3 == 0:
        if termination or truncation:
            action = None
        else:
            action_mask = np.array(acts_array[0, int(step/3), :], dtype=np.int8)
            action = env.action_space(agent).sample(action_mask) #action_mask
            
    
    if i%3 == 1:
        if termination or truncation:
            action = None
        else:
            action_mask = np.array(acts_array[1, int(step/3), :], dtype=np.int8)
            action = env.action_space(agent).sample(action_mask) #action_mask
            
    
    if i%3 == 2:
        if termination or truncation:
            action = None
        else:
            action_mask = np.array(acts_array[2, int(step/3), :], dtype=np.int8)
            action = env.action_space(agent).sample(action_mask) #action_mask
            
    
    print("i=",i, "action_m", action_mask, " act=", action)
    env.step(action)
    i += 1
    step += 1

env.close()

