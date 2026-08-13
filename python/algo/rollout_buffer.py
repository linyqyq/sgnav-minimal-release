import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, horizon: int, img_shape, vec_dim: int, action_dim: int, device: str = "cuda"):
        self.horizon = horizon
        self.device = torch.device(device)

        H, W, C = img_shape
        self.img = np.zeros((horizon, C, H, W), dtype=np.float32)  # store as CHW
        self.vec = np.zeros((horizon, vec_dim), dtype=np.float32)
        self.actions = np.zeros((horizon, action_dim), dtype=np.float32)
        self.logp = np.zeros((horizon, 1), dtype=np.float32)
        self.rewards = np.zeros((horizon, 1), dtype=np.float32)
        self.dones = np.zeros((horizon, 1), dtype=np.float32)
        self.values = np.zeros((horizon, 1), dtype=np.float32)

        self.advantages = np.zeros((horizon, 1), dtype=np.float32)
        self.returns = np.zeros((horizon, 1), dtype=np.float32)

        self.t = 0

    def add(self, img_chw, vec, action, logp, reward, done, value):
        assert self.t < self.horizon
        self.img[self.t] = img_chw
        self.vec[self.t] = vec
        self.actions[self.t] = action
        self.logp[self.t] = logp
        self.rewards[self.t] = reward
        self.dones[self.t] = float(done)
        self.values[self.t] = value
        self.t += 1

    def compute_gae(self, last_value: float, gamma=0.99, lam=0.95):
        adv = 0.0
        for t in reversed(range(self.horizon)):
            next_value = last_value if t == self.horizon - 1 else self.values[t + 1, 0]
            next_non_terminal = 1.0 - self.dones[t, 0]
            delta = self.rewards[t, 0] + gamma * next_value * next_non_terminal - self.values[t, 0]
            adv = delta + gamma * lam * next_non_terminal * adv
            self.advantages[t, 0] = adv
        self.returns[:, 0] = self.advantages[:, 0] + self.values[:, 0]

    def get_tensors(self):
        # Normalize advantages for stability
        adv = self.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        img_t = torch.from_numpy(self.img).to(self.device)          # (T,C,H,W)
        vec_t = torch.from_numpy(self.vec).to(self.device)
        act_t = torch.from_numpy(self.actions).to(self.device)
        logp_t = torch.from_numpy(self.logp).to(self.device)
        ret_t = torch.from_numpy(self.returns).to(self.device)
        adv_t = torch.from_numpy(adv).to(self.device)
        val_t = torch.from_numpy(self.values).to(self.device)

        return img_t, vec_t, act_t, logp_t, ret_t, adv_t, val_t
