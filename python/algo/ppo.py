import torch
import torch.nn as nn
import torch.nn.functional as F


class PPO:
    def __init__(
        self,
        model: nn.Module,
        lr=3e-4,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        epochs=4,
        minibatch_size=256,
        target_kl=None,
        device="cuda",
    ):
        self.model = model
        self.device = torch.device(device)
        self.opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.target_kl = target_kl

    def update(self, img, vec, actions, old_logp, returns, advantages):
        """
        img: (T,C,H,W)
        vec: (T,V)
        actions: (T,A)  (tanh-squashed)
        old_logp: (T,1)
        returns: (T,1)
        advantages: (T,1)
        """
        T = img.shape[0]
        idx = torch.randperm(T, device=self.device)

        total_actor, total_critic, total_ent, total_kl = 0.0, 0.0, 0.0, 0.0
        n_updates = 0
        early_stop = False

        for epoch in range(self.epochs):
            for start in range(0, T, self.minibatch_size):
                mb = idx[start:start+self.minibatch_size]

                new_logp, v, entropy = self.model.evaluate_actions(img[mb], vec[mb], actions[mb])

                log_ratio = new_logp - old_logp[mb]
                ratio = torch.exp(log_ratio)
                approx_kl = ((ratio - 1.0) - log_ratio).mean()

                # PPO clipped objective
                adv = advantages[mb]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * adv
                actor_loss = -(torch.min(surr1, surr2)).mean()

                # Value loss
                critic_loss = F.mse_loss(v, returns[mb])

                ent_loss = -entropy.mean()

                loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * ent_loss

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()

                total_actor += float(actor_loss.detach().cpu())
                total_critic += float(critic_loss.detach().cpu())
                total_ent += float(entropy.detach().cpu().mean())
                total_kl += float(approx_kl.detach().cpu())
                n_updates += 1

                if self.target_kl is not None and approx_kl > self.target_kl:
                    early_stop = True
                    break

            if early_stop:
                break

        return {
            "actor_loss": total_actor / max(n_updates, 1),
            "critic_loss": total_critic / max(n_updates, 1),
            "entropy": total_ent / max(n_updates, 1),
            "approx_kl": total_kl / max(n_updates, 1),
            "ppo_epochs_used": epoch + 1,
            "kl_early_stop": early_stop,
        }
