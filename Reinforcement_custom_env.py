import cv2
import time
import pickle
import numpy as np
from PIL import Image
from matplotlib import style
import matplotlib.pyplot as plt

style.use("ggplot")

size = 10
episodes = 25000
mv_penalty = 1
enemy_penalty = 300
food_reward = 25
epsilon = 0.9
eps_decay = 0.9998
render_at = 300
show = False

q_table = None

learning_rate = 0.1
discount = 0.95

player_n = 1
food_n = 2
enemy_n = 3

d = {1: (255, 175, 0),
	 2: (0, 255, 0),
	 3: (0, 0, 255)}

class Blob(object):
	def __init__(self):
		self.x = np.random.randint(0, size)
		self.y = np.random.randint(0, size)

	def __str__(self):
		return (f"{self.x}, {self.y}")
	
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
		if not x:
			self.x += np.random.randint(-1,2)
		else:
			self.x += x

		if not y:
			self.y += np.random.randint(-1,2)
		else:
			self.y += y

		if self.x < 0:
			self.x = 0
		elif self.x > size-1:
			self.x = size-1

		if self.y < 0:
			self.y = 0
		elif self.y > size-1:
			self.y = size-1

if q_table == None:
	q_table = {}
	for x1 in range(((-size) + 1), size):
		for y1 in range(((-size) + 1), size):
			for x2 in range(((-size) + 1), size):
				for y2 in range(((-size) + 1), size):
					q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
	with open(q_table, "rb") as f:
		q_table = pickle.load(f)

ep_rewards = []
for episode in range(episodes):
	player = Blob()
	food = Blob()
	enemy = Blob()

	if episode % render_at == 0:
		print(f"on # {episode}, epsilon: {epsilon}")
		print(f"{render_at} ep mean {np.mean(ep_rewards[-render_at:])}")
		show = True

	else:
		show = False

	ep_reward = 0
	for i in range(250):
		obs = (player - food, player - enemy)

		if np.random.random() > epsilon:
			action = np.argmax(q_table[obs])
		else:
			action = np.random.randint(0, 4)

		player.action(action)

		#####################
		#enemy.move()
		#food.move()
		#####################

		if player.x == enemy.x and player.y == enemy.y:
			reward = -enemy_penalty

		elif player.x == food.x and player.y == food.y:
			reward = food_reward

		else:
			reward = -mv_penalty

		new_obs = (player - food, player - enemy)
		max_future_q = np.max(q_table[new_obs])
		current_q = q_table[obs][action]

		if reward == food_reward:
			new_q = food_reward
		elif reward == -enemy_penalty:
			new_q = -enemy_penalty
		else:
			new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)

		q_table[obs][action] = new_q

		if show:
			env = np.zeros((size, size, 3), dtype = np.uint8)

			env[food.y][food.x] = d[food_n]
			env[player.y][player.x] = d[player_n]
			env[enemy.y][enemy.x] = d[enemy_n]

			img = Image.fromarray(env, "RGB")
			img = img.resize((300, 300))
			cv2.imshow("", np.array(img))			
			if reward == food_reward or reward == -enemy_penalty:
				if cv2.waitKey(500) & 0xff == ord("q"):
					break
			else:
				if cv2.waitKey(1) & 0xff == ord("q"):
					break

		ep_reward += reward
		if reward == food_reward or reward == -enemy_penalty:
			break

	ep_rewards.append(ep_reward)
	epsilon *= eps_decay

moving_avg = np.convolve(ep_rewards, np.ones((render_at,)) / render_at, mode = "valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward at {render_at}")
plt.xlabel("episode #")
plt.show()

with open(f"q_table-{int(time.time())}.pickle", "wb") as f:
	pickle.dump(q_table, f)