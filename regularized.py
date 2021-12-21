from google.colab import drive
in_drive = r"/content/drive/MyDrive/aim_mini/regulated_models"
drive.mount('/content/drive')
print("\n\n")

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
!pip install torchinfo
from torchinfo import summary
import time
from tqdm import tqdm
from copy import deepcopy

import random, numpy, os

from collections import namedtuple
!pip install git+https://github.com/kenjyoung/MinAtar
!pip install backpack-for-pytorch
!pip install gpytorch
!pip install git+https://github.com/AlexImmer/BNN-predictions
from minatar import Environment

os.chdir(r"/content/")
from images import show_rewards
from get_memorable import select_memorable_points



BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
TARGET_NETWORK_UPDATE_FREQ = 1000
TRAINING_FREQ = 1
NUM_FRAMES = 5000000
FIRST_N_FRAMES = 100000
REPLAY_START_SIZE = 5000
END_EPSILON = 0.1
STEP_SIZE = 0.00025
GRAD_MOMENTUM = 0.95
SQUARED_GRAD_MOMENTUM = 0.95
MIN_SQUARED_GRAD = 0.01
GAMMA = 0.99
EPSILON = 1.0
num_channels = 6

#import torch_xla_py.xla_model as xm
#device = xm.xla_device()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

game_names = ["asterix",
         "breakout",
         "space_invaders"]



class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(QNetwork, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        
        example = torch.zeros((1, num_channels, 10, 10))
        example = self.conv(example).flatten(1)
        quantity = example.shape[-1]
        
        self.fc_hidden = nn.Linear(in_features = quantity, out_features = 128)
        self.output = nn.Linear(in_features=128, out_features=num_actions)

    def forward(self, x):
        x = f.relu(self.conv(x))
        x = f.relu(self.fc_hidden(x.view(x.size(0), -1)))
        return self.output(x)
    
    
    
### My memory stuff!


transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []
        self.unchecked = []

    def add(self, *args):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)
        self.location = (self.location + 1) % self.buffer_size
        
        self.unchecked.append(transition(*args))
        
    def checked(self):
        self.unchecked = []
 
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


SIGMA = .1
NUM_POINTS = 6 * 50
memorable = {game : [] for game in game_names}
games_memorable = []

fromp = True

def get_mem(replay_buffer, model):
  #try:
    with torch.no_grad():
        batch_samples = transition(*zip(*replay_buffer.unchecked))
        replay_buffer.checked()
        if(fromp):
          states = torch.cat(batch_samples.state)
          actions = torch.cat(batch_samples.action)
          #print('states', states,'\n', 'actions', actions.squeeze(1))
          dataloader = [(states, actions.squeeze(1))]
          output = select_memorable_points(dataloader, model, num_points=NUM_POINTS, num_classes=6,
                                  use_cuda=False, label_set=None, descending=True)
          r_points = output[0]
        else:
          try:    r_points = random.sample(batch_samples.state, NUM_POINTS)
          except: r_points = batch_samples.state
          r_points = torch.cat(r_points, dim=0)

        out = model(r_points)
        for i in range(out.shape[0]):
          games_memorable.append([r_points[i], out[i]])    
  #except: pass

def get_full_mem(games_memorable, model, game):
    with torch.no_grad():
        if(fromp):
          states = torch.cat([games_memorable[i][0].unsqueeze(0) for i in range(len(games_memorable))])
          actions = model(states).max(1)[1]
          dataloader = [(states, actions)]
          output = select_memorable_points(dataloader, model, num_points = NUM_POINTS*10, num_classes=6,
                                  use_cuda=False, label_set=None, descending=True)
          r_points = output[0]
        else:
          try:    r_points = random.sample([games_memorable[i][0].unsqueeze(0) for i in range(len(games_memorable))], NUM_POINTS*10)
          except: r_points = [games_memorable[i][0].unsqueeze(0) for i in range(len(games_memorable))]
          r_points = torch.cat(r_points, dim=0)
        
        out = model(r_points)
        memorable[game] = (r_points, out)

mem_batch = 256
    
def reg(model, memory):
    indexes = [i for i in range(len(memory[0]))]
    random.shuffle(indexes)
    indexes = indexes[:mem_batch]
    remembered_states =  memory[0][indexes]
    remembered_actions = memory[1][indexes]
    model.eval()
    f_t = model(remembered_states)   
    f_t_m_1 = remembered_actions
    F = f_t - f_t_m_1
    reg = F**2 
    reg = SIGMA * torch.sum(reg)
    return(reg)

###



def get_state(s):
    s = torch.tensor(s)
    if(s.shape[-1] < num_channels):
        s = torch.cat([s, torch.ones(s.shape[:-1] + (num_channels - s.shape[-1],)) < .5], dim = -1)
    return s.permute(2, 0, 1).unsqueeze(0).float().to(device)



def world_dynamics(
        t, 
        replay_start_size, 
        num_actions, 
        s, 
        env, 
        policy_net):
    
    policy_net.train()

    if t < replay_start_size:
        action = torch.tensor([[random.randrange(num_actions)]], device=device)
    else:
        epsilon = END_EPSILON if t - replay_start_size >= FIRST_N_FRAMES \
            else ((END_EPSILON - EPSILON) / FIRST_N_FRAMES) * (t - replay_start_size) + EPSILON
        if numpy.random.binomial(1, epsilon) == 1:
            action = torch.tensor([[random.randrange(num_actions)]], device=device)
        else:
            with torch.no_grad():
                action = policy_net(s).max(1)[1].view(1, 1)
    reward, terminated = env.act(action)
    s_prime = get_state(env.state())
    return s_prime, action, torch.tensor([[reward]], device=device).float(), torch.tensor([[terminated]], device=device)



def train(
        sample, 
        policy_net, 
        target_net, 
        optimizer):
    
    policy_net.train()
    target_net.train()
    
    batch_samples = transition(*zip(*sample))

    states = torch.cat(batch_samples.state)
    next_states = torch.cat(batch_samples.next_state)
    actions = torch.cat(batch_samples.action)
    rewards = torch.cat(batch_samples.reward)
    is_terminal = torch.cat(batch_samples.is_terminal)
    
    Q_s_a = policy_net(states).gather(1, actions)
    
    none_terminal_next_state_index = torch.tensor([i for i, is_term in enumerate(is_terminal) if is_term == 0], dtype=torch.int64, device=device)
    none_terminal_next_states = next_states.index_select(0, none_terminal_next_state_index)
    Q_s_prime_a_prime = torch.zeros(len(sample), 1, device=device)
    if len(none_terminal_next_states) != 0:
        Q_s_prime_a_prime[none_terminal_next_state_index] = target_net(none_terminal_next_states).detach().max(1)[0].unsqueeze(1)

    target = rewards + GAMMA * Q_s_prime_a_prime
    loss = f.smooth_l1_loss(target, Q_s_a)
    # FROMP regularization!
    for game in memorable.keys():
        if(len(memorable[game]) > 0):
          loss += reg(policy_net, memorable[game])
    # For unregularized, comment out that for-loop.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  

def dqn(
        policy_net,
        target_net,
        game, 
        replay_off, 
        target_off, 
        output_file_name, 
        store_intermediate_result=False, 
        load_path=None, 
        step_size=STEP_SIZE, 
        time_limit = 60 * 60 * 6,
        start_time = 0):
  
    global games_memorable
    global memorable
    
    env = Environment(game)
    
    policy_net.train()
    if not target_off:
        target_net.train()
    replay_start_size = 0
    if not replay_off:
        r_buffer = replay_buffer(REPLAY_BUFFER_SIZE)
        replay_start_size = REPLAY_START_SIZE

    optimizer = optim.RMSprop(policy_net.parameters(), lr=step_size, alpha=SQUARED_GRAD_MOMENTUM, centered=True, eps=MIN_SQUARED_GRAD)

    e_init = 0
    t_init = 0
    policy_net_update_counter_init = 0

    t = t_init
    e = e_init
    policy_net_update_counter = policy_net_update_counter_init
    games = 0
    train_rewards = []
     
    program_starts = time.time()
    sec = start_time
    
    for t in tqdm(range(NUM_FRAMES)): 
        sec = time.time() - program_starts + start_time
        if(sec > time_limit): print("\n\nTime limit! Final result:\n\n"); break
        #print("{} seconds out of {}.".format(sec, time_limit))

        G = 0.0

        env.reset()
        games += 1
        total_rew = 0
        period = 5000
        s = get_state(env.state())
        is_terminated = False
        while(not is_terminated) and t < NUM_FRAMES:
            s_prime, action, reward, is_terminated = world_dynamics(t, replay_start_size, 6, s, env, policy_net)
            total_rew += reward.item()
            #if(games % 100 == 0): env.display_state()
            sample = None
            if replay_off:
                sample = [transition(s, s_prime, action, reward, is_terminated)]
            else:
                r_buffer.add(s, s_prime, action, reward, is_terminated)
                if t > REPLAY_START_SIZE and len(r_buffer.buffer) >= BATCH_SIZE:
                    sample = r_buffer.sample(BATCH_SIZE)

            if t % TRAINING_FREQ == 0 and sample is not None:
                if target_off:
                    train(sample, policy_net, policy_net, optimizer)
                else:
                    policy_net_update_counter += 1
                    train(sample, policy_net, target_net, optimizer)

            if not target_off and policy_net_update_counter > 0 and policy_net_update_counter % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            G += reward.item()
            t += 1
            s = s_prime
            
        train_rewards.append(total_rew)
        e += 1
        
        if(games % 1000 == 0): 
            print("{} games. {} / {}".format(games, t, NUM_FRAMES))
            for filename in os.listdir(in_drive):
                if(game in filename): os.remove(in_drive + "/" + filename)
            show_rewards("{}_regulated_at_sigma_{}_{}".format(game, SIGMA, str(round(sec)).zfill(5)), train_rewards)
            torch.save(policy_net.state_dict(), in_drive + r"/{}_regulated_at_sigma_{}_{}.pt".format(game, SIGMA, str(round(sec)).zfill(5)))
            get_mem(r_buffer, policy_net)
            print("\nLength of memorable in this game: {}.\nLength of other memerable: {}.".format(len(games_memorable), [memorable[game_name][0].shape[0] if(type(memorable[game_name]) != list) else 0 for game_name in game_names]))
            torch.save(memorable, in_drive + r"/{}_regulated_memory_at_sigma_{}_{}.pt".format(game, SIGMA, str(round(sec)).zfill(5)))
            torch.save(games_memorable, in_drive + r"/{}_regulated_game_memory_at_sigma_{}_{}.pt".format(game, SIGMA, str(round(sec)).zfill(5)))
    for filename in os.listdir(in_drive):
        if(game in filename): os.remove(in_drive + "/" + filename)
    show_rewards("{}_regulated_{}".format(game, str(round(sec)).zfill(5)), train_rewards)
    torch.save(policy_net.state_dict(), in_drive + r"/{}_regulated_at_sigma_{}_{}.pt".format(game, SIGMA, str(round(sec)).zfill(5)))
    get_mem(r_buffer, policy_net)
    get_full_mem(games_memorable, policy_net, game)
    games_memorable = []
    print("Length of memorable in this game (emptied): {}.\nLength of total memerable: {}.\n\n".format(len(games_memorable), [memorable[game_name][0].shape[0] if(type(memorable[game_name]) != list) else 0 for game_name in game_names]))
    torch.save(memorable, in_drive + r"/{}_regulated_memory_at_sigma_{}_{}.pt".format(game, SIGMA, str(round(sec)).zfill(5)))
    torch.save(games_memorable, in_drive + r"/{}_regulated_game_memory_at_sigma_{}_{}.pt".format(game, SIGMA, str(round(sec)).zfill(5)))


def main():
    global games_memorable
    global memorable

    output = None
    replayoff = False
    alpha = .00025 
    save = False
    targetoff = False  
    load_file_path = None
    file_name = output   

    policy_net = QNetwork(num_channels, 6).to(device)
    start_time = 0
    first_game = 0
    how_much_time = 60 * 60 * 1

    load_model = False # "/content/drive/MyDrive/aim_mini/regulated_models/asterix_regulated_at_sigma_1_03600.pt"
    if(load_model != False):
        policy_net.load_state_dict(torch.load(load_model))
        memorable = torch.load(load_model[:-19] + "memory_" + load_model[-19:])
        games_memorable = torch.load(load_model[:-19] + "game_memory_" + load_model[-19:])
        start_time = int(load_model[-8:-3])
        game_name = load_model[49:54]
        print("\n\n", game_name, "\n\n")
        if(game_name == "aster"): pass 
        elif(game_name == "break"): first_game = 1
        elif(game_name == "space"): first_game = 2
        if(start_time >= how_much_time): start_time = 0; first_game += 1
        print("\n\n\nStarting with {}, {} seconds in.\n\n\n".format(policy_net, start_time))
    target_net = QNetwork(num_channels, 6).to(device)
    target_net.load_state_dict(policy_net.state_dict()) 
     
    print("\n\n{}\n\n\n{}\n\n".format(policy_net, summary(policy_net, (1, num_channels, 10, 10))))  
     
    for game in game_names[first_game:]:  
        dqn(policy_net = policy_net,  
            target_net = target_net, 
            game = game,   
            replay_off = replayoff,   
            target_off = targetoff,  
            output_file_name = file_name,
            store_intermediate_result = save, 
            load_path = load_file_path, 
            step_size = alpha, 
            time_limit = how_much_time, 
            start_time = start_time)       
        start_time = 0   

if __name__ == '__main__':  
    main()
