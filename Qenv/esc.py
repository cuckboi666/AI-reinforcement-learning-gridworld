import numpy as np 
from PIL import Image 
import cv2 
import matplotlib.pyplot as plt 
import pickle 
from matplotlib import style 
import time 


style.use("ggplot") 

SIZE = 10 
HM_EPISODES = 50000
MOVE_PENALTY = 1 
ENEMY_PENALTY = 300
FOOD_REWARD = 25 
epsilon = 1.0
EPS_DECAY = 0.9998 
SHOW_EVERY = 5000

start_q_table = None # or that file name for existing Q table 

LEARNING_RATE = 0.1
DISCOUNT = 0.95 

PLAYER_N = 1 
FOOD_N = 2 
ENEMY_N = 3 

d = {1: (255,175,0), # player_n color
     2: (0, 255, 0), # food_n color 
     3: (0, 0, 255)} # enemy_n color 


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
    
    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other): 
        return (self.x - other.x, self.y - other.y) 

    def action(self, choice): 
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
         

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2) 
        else:
            self.x += x 
        
        # If no value for y, move randomly 
        if not y:
            self.y += np.random.randint(-1, 2) 
        else:
            self.y += y

        # If Out of bounds..... FIX!
        if self.x < 0:
            self.x = 0 
        elif self.x > SIZE - 1:
            self.x = SIZE - 1

        if self.y < 0:
            self.y = 0 
        elif self.y > SIZE - 1:
            self.y = SIZE - 1

if start_q_table is None:
    # initialize the q-table#
    q_table = {} 
    for i in range(-SIZE+1, SIZE):
        for ii in range(-SIZE+1, SIZE):
            for iii in range(-SIZE+1, SIZE):
                for iiii in range(-SIZE+1, SIZE):
                    q_table[((i, ii), (iii, iiii))] = [np.random.uniform(-5,0) for i in range(4)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f) 

episode_rewards = [] 

for episode in range(HM_EPISODES):
    player = Blob() 
    food = Blob() 
    enemy = Blob() 

    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}") 
        show = True 
    else:
        show = False 

    episode_reward = 0 
    for i in range(200): 
        obs = (player-food, player-enemy)
        # print the obs
        if np.random.random() > epsilon:
            # get dat action now! 
            action = np.argmax(q_table[obs]) 
        else:
            action = np.random.randint(0, 4) 
        # TAKE ACTION!
        player.action(action) 


        ##### MAYBE LATER 
        
        enemy.move() 
        food.move() 
        
        # let the enemy move and eat shit maybe? 
        #############


        if player.x == enemy.x and player.y == enemy.y: 
            reward = -ENEMY_PENALTY 
        
        elif player.x == food.x and player.y == food.y: 
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

            # Know the reward, time for calculatioon
            # obs immediately after move

        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs]) 
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD: 
            new_q = FOOD_REWARD 
        
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q 

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8) 
            env[food.y][food.x] =  d[FOOD_N]
            env[player.y][player.x] =  d[PLAYER_N]
            env[enemy.y][enemy.x] =  d[ENEMY_N]

            img = Image.fromarray(env, "RGB") 
            img = img.resize((300, 300)) 
            cv2.imshow("image", np.array(img)) 
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY: 
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break 

        episode_reward += reward 
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break 
    
    episode_rewards.append(episode_reward) 
    epsilon *= EPS_DECAY 

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg) 
plt.ylabel(f"reward {SHOW_EVERY}ma") 
plt.xlabel("episode #") 
plt.show() 

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
    
    
             
            






































































































































































