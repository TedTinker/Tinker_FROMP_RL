import matplotlib.pyplot as plt
import numpy as np
from minatar import Environment
import matplotlib

    
    


def rolling(rewards, dist = 500):
    rolled = []
    for i in range(len(rewards)):
        if(i+1 <= dist):   
            rolled.append(sum(rewards[:i+1]) / len(rewards[:i+1]))
        else:       
            rolled.append(sum(rewards[i - dist + 1 : i+1]) / dist)
    return(rolled)



def show_rewards(title, train_rewards, test_rewards = None, quantity = 15000):
    if(test_rewards == None): test_rewards = train_rewards
    t = [i for i in range(len(train_rewards))]
    rolled_train = rolling(train_rewards)
    rolled_test = rolling(test_rewards)
    if(len(train_rewards) > quantity):
        train_rewards = train_rewards[-quantity:]
        rolled_train = rolled_train[-quantity:]
        test_rewards = test_rewards[-quantity:]
        rolled_test = rolled_test[-quantity:]
        t = t[-quantity:]
    #matplotlib.use("module://matplotlib_inline.backend_inline")
    #with plt.ioff():
    plt.title(title)
    plt.plot(t, train_rewards, color = "pink")
    plt.plot(t, test_rewards, color = "aqua")
    plt.plot(t, rolled_train, color = "red")
    plt.plot(t, rolled_test, color = "blue")
    #plt.savefig("/content/drive/MyDrive/aim_mini/regulated_plots/" + title + '.png')
    plt.show()
    plt.close()
    #matplotlib.use('TkAgg')