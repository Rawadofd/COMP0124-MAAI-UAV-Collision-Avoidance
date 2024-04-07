import numpy as np
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
    
def loss(file_path):
    import numpy as np
    import re
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    avg_reward = []
    
    with open(file_path, 'rb') as f:
        event_acc = EventAccumulator(f.name)
        event_acc.Reload()
        tags = event_acc.Tags()
        
        print(tags)

        # Load the loss
        tag = 'policy_0/loss'
        loss = event_acc.Scalars(tag)
        loss_value = []
        for i in loss:
            print(i.step, i.value)
            loss_value.append(i.value)
        
        # Load the average episode reward
        tag2 = 'average_episode_reward'
        reward = event_acc.Scalars(tag2)
        for r in reward:
            avg_reward.append(r.value)     
        
        
    return loss_value

# file_path = 'D:/MAAI/off-policy/offpolicy/scripts/results/MPE/simple_spread/qmix/debug/run39/logs/policy_0/loss/policy_0/loss/events.out.tfevents.1712437997.DESKTOP-U9OC6U6'
# loss_value = loss(file_path)

# iter = range(len(loss_value))
# plt.plot(iter, loss_value, '-', color='blue')  
# plt.title('Loss Curve')  
# plt.show()



def average_reward(file_path):
    import numpy as np
    import re
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    avg_reward = []
    
    with open(file_path, 'rb') as f:
        event_acc = EventAccumulator(f.name)
        # Load the log file
        event_acc.Reload()
        tags = event_acc.Tags()
        print(tags)
        
        # Load the average episode reward
        tag2 = 'average_episode_rewards'
        reward = event_acc.Scalars(tag2)
        for r in reward:
            avg_reward.append(r.value)     
        
        
    return avg_reward

file_path2 = 'D:/MAAI/off-policy/offpolicy/scripts/results/MPE/simple_spread/qmix/debug/run39/logs/average_episode_rewards/average_episode_rewards/events.out.tfevents.1712437997.DESKTOP-U9OC6U6'
avg_reward = average_reward(file_path2)

iter = range(len(avg_reward))
plt.plot(iter, avg_reward, '-', color='blue')  
plt.title('Average Episode Reward')  
plt.show()