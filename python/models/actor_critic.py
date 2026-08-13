import torch
import torch.nn as nn
from torch.distributions import Normal

from .clip_encoder import ClipVisionEncoder


def mlp(in_dim, hidden_dims, out_dim, act=nn.Tanh):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """
    PPO Actor-Critic with dual-branch text-guided CLIP:
      - target branch: focus on blue target
      - obstacle branch: focus on buoys / obstacles
    """
    def __init__(
        self,
        action_dim: int,
        vec_dim: int = 27,
        use_clip: bool = False,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        freeze_clip: bool = True,
        proj_dim: int = 128,
        trunk_dim: int = 256,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.use_clip = use_clip
        self.proj_dim = proj_dim
        self.obstacle_proj_dim = proj_dim // 2
        self.freeze_clip = freeze_clip

        # ===== target / obstacle / background prompts =====
        self.target_fg_prompts = [
            "magenta sphere",
            "small magenta sphere",
            "round magenta object",
        ]

        self.obstacle_fg_prompts = [
            "green navigation buoy",
            "red navigation buoy",
            "black navigation buoy",
            "small vertical buoy on water",
            "channel marker buoy",
        ]

        self.bg_prompts = [
            "sky",
            "cloud",
            "sea surface",
            "ocean water",
            "horizon",
        ]

        # stronger guidance for target, weaker for obstacle
        self.target_guidance_lambda = 2.0
        self.obstacle_guidance_lambda = 1.0

        if self.use_clip:
            self.vision = ClipVisionEncoder(
                model_name=clip_model,
                pretrained=clip_pretrained,
                device=device,
                freeze=freeze_clip,
            )

            # main branch for navigation target
            self.target_img_proj = nn.Sequential(
                nn.Linear(self.vision.embed_dim, proj_dim),
                nn.ReLU(),
                nn.LayerNorm(proj_dim),
            )

            # lighter branch for obstacles
            self.obstacle_img_proj = nn.Sequential(
                nn.Linear(self.vision.embed_dim, self.obstacle_proj_dim),
                nn.ReLU(),
                nn.LayerNorm(self.obstacle_proj_dim),
            )

            print("[ActorCritic] Dual-branch CLIP enabled")
            print("  clip_model:", clip_model)
            print("  clip_pretrained:", clip_pretrained)
            print("  target_fg_prompts:", self.target_fg_prompts)
            print("  obstacle_fg_prompts:", self.obstacle_fg_prompts)
            print("  bg_prompts:", self.bg_prompts)
            print("  target_guidance_lambda:", self.target_guidance_lambda)
            print("  obstacle_guidance_lambda:", self.obstacle_guidance_lambda)
            print("  freeze_clip:", self.freeze_clip)
        else:
            self.vision = None
            self.target_img_proj = None
            self.obstacle_img_proj = None

        # target feature + obstacle feature + vector state
        self.trunk = nn.Sequential(
            nn.Linear(proj_dim + self.obstacle_proj_dim + vec_dim, trunk_dim),
            nn.Tanh(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.Tanh(),
        )

        self.actor_mu = nn.Linear(trunk_dim, action_dim)
        self.critic_v = nn.Linear(trunk_dim, 1)

        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.to(self.device)

    def forward(self, img_bchw: torch.Tensor, vec: torch.Tensor):
        """
        img_bchw: (B,3,H,W)
        vec: (B,vec_dim)
        returns: mu (B,A), value (B,1)
        """
        vec = vec.to(self.device).float()

        if self.use_clip:
            with torch.no_grad():
                # ===== target branch =====
                target_emb, target_weights, _, target_fg_scores, target_bg_scores = (
                    self.vision.forward_text_guided(
                        img_bchw,
                        fg_prompts=self.target_fg_prompts,
                        bg_prompts=self.bg_prompts,
                        text_guidance_lambda=self.target_guidance_lambda,
                        temperature=0.2,
                        use_spatial_prior=True,
                    )
                )

                # ===== obstacle branch =====
                obstacle_emb, obstacle_weights, _, obstacle_fg_scores, obstacle_bg_scores = (
                    self.vision.forward_text_guided(
                        img_bchw,
                        fg_prompts=self.obstacle_fg_prompts,
                        bg_prompts=self.bg_prompts,
                        text_guidance_lambda=self.obstacle_guidance_lambda,
                        temperature=0.2,
                        use_spatial_prior=True,
                    )
                )

            z_target = self.target_img_proj(target_emb)
            z_obstacle = self.obstacle_img_proj(obstacle_emb)

            # ===== cache debug info =====
            self.last_clip_debug = {
                "target_weights_mean": float(target_weights.mean().detach().cpu().item()),
                "target_weights_std": float(target_weights.std().detach().cpu().item()),
                "target_weights_min": float(target_weights.min().detach().cpu().item()),
                "target_weights_max": float(target_weights.max().detach().cpu().item()),
                "target_fg_mean": float(target_fg_scores.mean().detach().cpu().item()),
                "target_bg_mean": float(target_bg_scores.mean().detach().cpu().item()),

                "obstacle_weights_mean": float(obstacle_weights.mean().detach().cpu().item()),
                "obstacle_weights_std": float(obstacle_weights.std().detach().cpu().item()),
                "obstacle_weights_min": float(obstacle_weights.min().detach().cpu().item()),
                "obstacle_weights_max": float(obstacle_weights.max().detach().cpu().item()),
                "obstacle_fg_mean": float(obstacle_fg_scores.mean().detach().cpu().item()),
                "obstacle_bg_mean": float(obstacle_bg_scores.mean().detach().cpu().item()),
            }

            if torch.rand(1).item() < 0.01:
                print("====== DUAL CLIP DEBUG ======")
                print("target stats:", {
                    "w_std": self.last_clip_debug["target_weights_std"],
                    "w_max": self.last_clip_debug["target_weights_max"],
                    "fg": self.last_clip_debug["target_fg_mean"],
                    "bg": self.last_clip_debug["target_bg_mean"],
                })
                print("obstacle stats:", {
                    "w_std": self.last_clip_debug["obstacle_weights_std"],
                    "w_max": self.last_clip_debug["obstacle_weights_max"],
                    "fg": self.last_clip_debug["obstacle_fg_mean"],
                    "bg": self.last_clip_debug["obstacle_bg_mean"],
                })
        else:
            z_target = torch.zeros((vec.shape[0], self.proj_dim), device=self.device, dtype=vec.dtype)
            z_obstacle = torch.zeros((vec.shape[0], self.obstacle_proj_dim), device=self.device, dtype=vec.dtype)

        z = torch.cat([z_target, z_obstacle, vec], dim=-1)
        h = self.trunk(z)

        mu = self.actor_mu(h)
        v = self.critic_v(h)
        return mu, v

    def get_dist(self, mu: torch.Tensor):
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)
        return Normal(mu, std)

    def act(self, img_bchw: torch.Tensor, vec: torch.Tensor, deterministic: bool = False):
        mu, v = self.forward(img_bchw, vec)
        dist = self.get_dist(mu)

        if deterministic:
            a = mu
        else:
            a = dist.sample()

        logp = dist.log_prob(a).sum(-1, keepdim=True)
        a_tanh = torch.tanh(a)
        return a_tanh, logp, v

    def evaluate_actions(self, img_bchw: torch.Tensor, vec: torch.Tensor, actions_tanh: torch.Tensor):
        mu, v = self.forward(img_bchw, vec)
        dist = self.get_dist(mu)

        eps = 1e-6
        a = torch.clamp(actions_tanh, -1 + eps, 1 - eps)
        a_pre = 0.5 * (torch.log1p(a) - torch.log1p(-a))

        logp = dist.log_prob(a_pre).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return logp, v, entropy
