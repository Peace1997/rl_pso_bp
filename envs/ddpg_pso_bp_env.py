from pso_bp.pso_bp import PSO_BP
import numpy as np
import gym
from gym import spaces


class ENV:
    def __init__(self) -> None:
        self.observation_space = spaces.Box(low=0.0, high=100, shape=(5,))
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,))

    def reset(
        self,
    ):
        self.num_particle = 30
        self.input_dim = 15
        self.output_dim = 1
        self.hidden_dim = 32
        self.sum_dim = (
            (self.hidden_dim * self.input_dim)
            + self.hidden_dim
            + (self.hidden_dim * self.output_dim)
            + self.output_dim
        )
        self.x_range = [-1, 1]
        self.v_range = [-0.5, 0.5]

        self.w_max = 0.9
        self.w_min = 0.4
        self.max_iter = 300
        self.iter = 0
        self.min_fitness = 0.002
        self.w = 0.5
        self.c1 = 2
        self.c2 = 2
        self.env = PSO_BP(
            self.num_particle,
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
            self.sum_dim,
            self.x_range,
            self.v_range,
            self.w_max,
            self.w_min,
            self.w,
            self.c1,
            self.c2,
            self.max_iter,
            self.min_fitness,
        )
        obs = self.env.final_compute_one_obs(self.w, self.c1, self.c2)
        self.pre_gbest_loss = self.env.Gbest_loss
        self.pre_mean_diverse = obs[2]
        self.pre_action = [0.0, 0.0, 0.0]

        return obs

    def step(self, action):
        action = (action + 1) / 2
        reward = 0
        done = 0
        w, c1, c2 = (
            np.clip(action[0], 0.4, 0.9),
            np.clip(action[1], 0, 1) * 3,
            np.clip(action[2], 0, 1) * 3,
        )

        obs = self.env.final_compute_one_obs(w, c1, c2)
        mean_diverse = obs[2]

        for i in range(3):
            reward -= abs(action[i] - self.pre_action[i]) / 100
        self.pre_action = action


        # if self.iter<self.max_iter/3 and self.env.Gbest_loss < self.pre_gbest_loss and mean_diverse > self.pre_mean_diverse:
        #     reward +=2
        # elif self.iter<self.max_iter/3 and self.env.Gbest_loss < self.pre_gbest_loss and mean_diverse <= self.pre_mean_diverse:
        #     reward +=1
        # elif self.iter<self.max_iter/3 and self.env.Gbest_loss > self.pre_gbest_loss and mean_diverse > self.pre_mean_diverse:
        #     reward -=1
        # elif self.iter<self.max_iter/3 and self.env.Gbest_loss > self.pre_gbest_loss and mean_diverse <= self.pre_mean_diverse:
        #     reward -=2
        # elif self.iter<self.max_iter*2/3 and self.env.Gbest_loss < self.pre_gbest_loss:
        #     reward +=1
        # elif self.iter<self.max_iter*2/3 and self.env.Gbest_loss >= self.pre_gbest_loss:
        #     reward -=1
        # elif self.iter<self.max_iter and self.env.Gbest_loss < self.pre_gbest_loss and mean_diverse <= self.pre_mean_diverse:
        #     reward +=2
        # elif self.iter<self.max_iter and self.env.Gbest_loss < self.pre_gbest_loss and mean_diverse > self.pre_mean_diverse:
        #     reward +=1
        # elif self.iter<self.max_iter and self.env.Gbest_loss > self.pre_gbest_loss and mean_diverse <= self.pre_mean_diverse:
        #     reward -=1
        # elif self.iter<self.max_iter and self.env.Gbest_loss < self.pre_gbest_loss and mean_diverse > self.pre_mean_diverse:
        #     reward -=2

        # if self.env.Gbest_loss < self.pre_gbest_loss and mean_diverse > self.pre_mean_diverse :
        #     reward += (self.pre_gbest_loss - self.env.Gbest_loss ) *2000
        # elif self.env.Gbest_loss < self.pre_gbest_loss and mean_diverse <= self.pre_mean_diverse:
        #     reward += (self.pre_gbest_loss - self.env.Gbest_loss ) *1000
        # elif self.env.Gbest_loss >= self.pre_gbest_loss and mean_diverse > self.pre_mean_diverse:
        #     reward += (self.pre_gbest_loss - self.env.Gbest_loss ) *1000
        # else:
        #     reward += (self.pre_gbest_loss - self.env.Gbest_loss ) *2000
        # a =float(f"{self.pre_gbest_loss-self.env.Gbest_loss:.4f}")
        # print("a",a)
        # if (a):
        #     reward +=1
        # else:
        #     reward -=1
        # reward += (self.pre_gbest_loss - self.env.Gbest_loss) * 1000
        # reward -=0.1
        reward -= self.env.Gbest_loss - 0.001
        self.pre_mean_diverse = mean_diverse
        self.pre_gbest_loss = self.env.Gbest_loss
        self.env.iter += 1
        #
        # #reward -= obs[0]
        #
        #
        if self.env.Gbest_loss < self.min_fitness:
            done = 1
            reward += 100
        if self.env.iter > self.env.max_iter:
            done = 1
        info = {}
        print("iter num:", self.env.iter)
        print("obs:", obs)
        print("action:", action, w, c1, c2)
        print("fitness:", self.env.Gbest_loss)
        print("reward:", reward)
        return obs, reward, done, info


if __name__ == "__main__":
    env = ENV()
    obs = env.reset()
    obs, reward, done, info = env.step([1, 2])
