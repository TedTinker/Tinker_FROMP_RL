import torch
from torch import nn
import torch.nn.functional as f
from minatar import Environment
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(QNetwork, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        
        example = torch.zeros((1, 6, 10, 10))
        example = self.conv(example).flatten(1)
        quantity = example.shape[-1]
        
        self.fc_hidden = nn.Linear(in_features = quantity, out_features = 128)
        self.output = nn.Linear(in_features=128, out_features=num_actions)

    def forward(self, x):
        x = f.relu(self.conv(x))
        x = f.relu(self.fc_hidden(x.view(x.size(0), -1)))
        return self.output(x)

#folder = "models_frorp"
#folder = "models_fromp"
folder = "models_fromp_no_repeats"

    
file = r"C:\Users\tedjt\Desktop\OIST\Rotation_1\aim_mini" + "\\" + folder
os.chdir(file)

policy_net = QNetwork(6, 6).to(device)
policy_net.eval()

games = ["asterix",
         "breakout",
         "space_invaders"]

model_names = [
    "random", 
    "asterix_transfer.pt",
    "breakout_unreg.pt",
    "space_invaders_unreg.pt",
    "asterix_transfer.pt",
    "breakout_reg_0.1.pt",
    "space_invaders_reg_0.1.pt"
    ]

model_names_plus_space = [model_names[0]] + [""] + model_names[1:4] + [""] + model_names[4:]



def get_state(s):
    s = torch.tensor(s)
    if(s.shape[-1] < 6):
        s = torch.cat([s, torch.ones(s.shape[:-1] + (6 - s.shape[-1],)) < .5], dim = -1)
    return s.permute(2, 0, 1).unsqueeze(0).float().to(device)



def test(game, model_name = "random", render = False):
    #print('Testing "{}" on "{}"...'.format(model_name, game), end = "")
    if(model_name != "random"):
        policy_net.load_state_dict(torch.load(model_name))
        policy_net.eval()
    env = Environment(game)
    rew = 0
    done = False
    frames = 0
    while(not done):
        if(render): env.display_state()
        s = get_state(env.state())
        if(model_name != "random"): action = policy_net(s).max(1)[1].view(1, 1)
        else: action = torch.tensor([[random.randrange(6)]], device=device)
        reward, done = env.act(action)
        rew += reward 
        frames += 1
    if(render): env.close_display()
    #print("\n\tDone!")
    return(rew)

def test_all_games(model_name = "random", render = False):
    rew = []
    for game in games:
        rew.append(test(game, model_name, render))
    return(rew)
        
def test_all_models(render = False):
    rew = []
    for name in model_names:
        rew.append(test_all_games(name, render))
    return(rew)

def test_repeatedly(repeats = 5):
    rew = np.zeros((repeats, len(model_names), len(games)))
    for i in range(repeats):
        print("{}... ".format(i), end = "")
        new_rew = np.array(test_all_models())
        rew[i] = new_rew
    std = np.std(rew, axis = 0)
    rew = np.sum(rew, axis = 0) / repeats
    print("Done!")
    return(rew, std)



rew, std = test_repeatedly(1000)

for i, name in enumerate(model_names_plus_space):
    if(name == ""):
        rew = np.insert(rew, i, 0, axis = 0)
        std = np.insert(std, i, 0, axis = 0)
    

matplotlib.use("module://matplotlib_inline.backend_inline")
width = .15
x = np.arange(len(model_names_plus_space))
for i in range(rew.shape[1]):
    #plt.fill_between( x = x + width*i + .15, y1 = rew[:,i] + std[:,i], y2 = rew[:,i] - std[:,i], color = "black", alpha = .5)
    plt.bar(x + width*i + .15, rew[:,i], width)
plt.xticks(x + width*2, model_names_plus_space, rotation=90)
plt.legend(games, loc = "upper left")
for i in range(rew.shape[0]):
    if(i in [1, 5, 8]):        pass
    elif(i in [0, 4]):          plt.axvline(x[i] + width*5.4 + .5, color='black', linestyle="-")
    else:                       plt.axvline(x[i] + width*5.4, color='black', linestyle= (0, (1, 5)))
plt.show()

from copy import deepcopy
normalized_rew = deepcopy(rew)
normalized_std = deepcopy(std)
for i in range(normalized_rew.shape[1]): normalized_rew[:,i] = rew[:,i] / rew[:,i].max()
for i in range(normalized_std.shape[1]): normalized_std[:,i] = std[:,i] / std[:,i].max()
x = np.arange(len(model_names_plus_space))
for i in range(normalized_rew.shape[1]):
    #plt.fill_between( x = x + width*i + .15, y1 = normalized_rew[:,i] + normalized_std[:,i], y2 = normalized_rew[:,i] - normalized_std[:,i], color = "black", alpha = .5)
    plt.bar(x + width*i + .15, normalized_rew[:,i], width)
plt.xticks(x + width*2, model_names_plus_space, rotation=90)
plt.legend(games, loc = "upper center")
for i in range(normalized_rew.shape[0]):
    if(i in [1, 5, 8]):        pass
    elif(i in [0, 4]):          plt.axvline(x[i] + width*5.4 + .5, color='black', linestyle="-")
    else:                       plt.axvline(x[i] + width*5.4, color='black', linestyle= (0, (1, 5)))
plt.ylim(0, 1.4)
plt.show()