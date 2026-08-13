# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Normal

# from .clip_encoder import ClipVisionEncoder


# def mlp(in_dim, hidden_dims, out_dim, act=nn.Tanh):
#     layers = []
#     prev = in_dim
#     for h in hidden_dims:
#         layers += [nn.Linear(prev, h), act()]
#         prev = h
#     layers += [nn.Linear(prev, out_dim)]
#     return nn.Sequential(*layers)


# class ActorCritic(nn.Module):
#     """
#     PPO Actor-Critic with CLIP vision encoder (frozen) + projection head.
#     """
#     def __init__(
#         self,
#         action_dim: int,
#         vec_dim: int = 27,
#         use_clip: bool = False,
#         clip_model: str = "ViT-B-32",
#         clip_pretrained: str = "openai",
#         proj_dim: int = 128,
#         trunk_dim: int = 256,
#         device: str = "cuda",
#     ):
#         super().__init__()
#         self.device = torch.device(device)
#         self.use_clip = use_clip
#         self.proj_dim = proj_dim

#         if self.use_clip:
#             self.vision = ClipVisionEncoder(model_name=clip_model, pretrained=clip_pretrained, device=device)
#             self.img_proj = nn.Sequential(
#                 nn.Linear(self.vision.embed_dim, proj_dim),
#                 nn.ReLU(),
#                 nn.LayerNorm(proj_dim),
#             )
#         else:
#             self.vision = None
#             self.img_proj = None

#         self.trunk = nn.Sequential(
#             nn.Linear(proj_dim + vec_dim, trunk_dim),
#             nn.Tanh(),
#             nn.Linear(trunk_dim, trunk_dim),
#             nn.Tanh(),
#         )

#         self.actor_mu = nn.Linear(trunk_dim, action_dim)
#         self.critic_v = nn.Linear(trunk_dim, 1)

#         # Global log_std (stable start). You can also make it state-dependent if needed.
#         self.log_std = nn.Parameter(torch.zeros(action_dim))

#         self.to(self.device)

#     def forward(self, img_bchw: torch.Tensor, vec: torch.Tensor):
#         """
#         img_bchw: (B,3,H,W)
#         vec: (B,vec_dim)
#         returns: mu (B,A), value (B,1)
#         """
#         vec = vec.to(self.device).float()

#         if self.use_clip:
#             with torch.no_grad():
#                 emb = self.vision(img_bchw)  # (B,D)
#             z_img = self.img_proj(emb)      # (B,proj_dim)
#         else:
#             # CLIP masked/off: use zero image features and rely on vector observations only.
#             z_img = torch.zeros((vec.shape[0], self.proj_dim), device=self.device, dtype=vec.dtype)

#         z = torch.cat([z_img, vec], dim=-1)
#         h = self.trunk(z)

#         mu = self.actor_mu(h)
#         v = self.critic_v(h)
#         return mu, v

#     def get_dist(self, mu: torch.Tensor):
#         std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)
#         return Normal(mu, std)

#     def act(self, img_bchw: torch.Tensor, vec: torch.Tensor):
#         mu, v = self.forward(img_bchw, vec)
#         dist = self.get_dist(mu)
#         a = dist.sample()
#         logp = dist.log_prob(a).sum(-1, keepdim=True)

#         # Squash to [-1,1] for Unity continuous actions (common). If your Unity expects raw, remove tanh.
#         a_tanh = torch.tanh(a)

#         # Note: If you use tanh squashing, the log-prob should include tanh correction.
#         # For simplicity (and often OK in practice), we omit correction here.
#         # If you want exactness, tell me and I’ll give you the corrected version.
#         return a_tanh, logp, v

#     def evaluate_actions(self, img_bchw: torch.Tensor, vec: torch.Tensor, actions_tanh: torch.Tensor):
#         """
#         Evaluate logprob/value/entropy for PPO update.
#         actions_tanh: actions after tanh (B,A)
#         """
#         mu, v = self.forward(img_bchw, vec)
#         dist = self.get_dist(mu)

#         # Inverse tanh (atanh) to map back to pre-squash space for approximate logprob
#         eps = 1e-6
#         a = torch.clamp(actions_tanh, -1 + eps, 1 - eps)
#         a_pre = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh

#         logp = dist.log_prob(a_pre).sum(-1, keepdim=True)
#         entropy = dist.entropy().sum(-1, keepdim=True)
#         return logp, v, entropy
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    PPO Actor-Critic with CLIP vision encoder (frozen) + projection head.
    """
    def __init__(
        self,
        action_dim: int,
        vec_dim: int = 27,
        use_clip: bool = False,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        fg_prompts=None,
        bg_prompts=None,
        text_guidance_lambda: float = 1.0,
        freeze_clip: bool = True,
        proj_dim: int = 128,
        trunk_dim: int = 256,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.use_clip = use_clip
        self.proj_dim = proj_dim

        # ===== text-guided CLIP config =====
        self.fg_prompts = fg_prompts or [
            "magenta target",
            "magenta sphere",
            "magenta ball",
            "green buoy",
            "black buoy",
            "floating navigation marker",
            "colored buoy on water",
            "small floating object",
        ]

        self.bg_prompts = bg_prompts or [
            "ocean",
            "sea surface",
            "water background",
            "open water",
            "waves",
            "water reflection",
            "sky",
            "horizon",
        ]
        self.text_guidance_lambda = text_guidance_lambda
        self.freeze_clip = freeze_clip

        if self.use_clip:
            self.vision = ClipVisionEncoder(
                model_name=clip_model,
                pretrained=clip_pretrained,
                device=device,
                # 如果 ClipVisionEncoder 还不支持 freeze 参数，这行先删掉
                freeze=freeze_clip,
            )
            self.img_proj = nn.Sequential(
                nn.Linear(self.vision.embed_dim, proj_dim),
                nn.ReLU(),
                nn.LayerNorm(proj_dim),
            )

            print("[ActorCritic] CLIP enabled")
            print("  clip_model:", clip_model)
            print("  clip_pretrained:", clip_pretrained)
            print("  fg_prompts:", self.fg_prompts)
            print("  bg_prompts:", self.bg_prompts)
            print("  text_guidance_lambda:", self.text_guidance_lambda)
            print("  freeze_clip:", self.freeze_clip)
        else:
            self.vision = None
            self.img_proj = None

        self.trunk = nn.Sequential(
            nn.Linear(proj_dim + vec_dim, trunk_dim),
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

        # if self.use_clip:
        #     with torch.no_grad():
        #         emb, weights, patch_tokens, fg_scores, bg_scores = self.vision.forward_text_guided(
        #             img_bchw,
        #             fg_prompts=self.fg_prompts,
        #             bg_prompts=self.bg_prompts,
        #             text_guidance_lambda=self.text_guidance_lambda,
        #         )

        #      # ===== DEBUG（加在这里）=====
        #     if torch.rand(1).item() < 0.01:
        #         print("====== CLIP DEBUG ======")
        #         print("weights:", weights[0][:5].detach().cpu().numpy())
        #         print("fg_scores:", fg_scores[0][:5].detach().cpu().numpy())
        #         print("bg_scores:", bg_scores[0][:5].detach().cpu().numpy())

        #     z_img = self.img_proj(emb)
        # else:
        #     z_img = torch.zeros((vec.shape[0], self.proj_dim), device=self.device, dtype=vec.dtype)
        if self.use_clip:
            with torch.no_grad():
                emb, weights, patch_tokens, fg_scores, bg_scores = self.vision.forward_text_guided(
                    img_bchw,
                    fg_prompts=self.fg_prompts,
                    bg_prompts=self.bg_prompts,
                    text_guidance_lambda=self.text_guidance_lambda,
                )

            # ===== 缓存训练时可读的 CLIP 统计信息 =====
            self.last_clip_debug = {
                "weights_mean": float(weights.mean().detach().cpu().item()),
                "weights_std": float(weights.std().detach().cpu().item()),
                "weights_min": float(weights.min().detach().cpu().item()),
                "weights_max": float(weights.max().detach().cpu().item()),
                "fg_mean": float(fg_scores.mean().detach().cpu().item()),
                "bg_mean": float(bg_scores.mean().detach().cpu().item()),
            }

            # ===== 可选：保留你现在这种小概率打印 =====
            if torch.rand(1).item() < 0.01:
                print("====== CLIP DEBUG ======")
                print("weights:", weights[0][:5].detach().cpu().numpy())
                print("fg_scores:", fg_scores[0][:5].detach().cpu().numpy())
                print("bg_scores:", bg_scores[0][:5].detach().cpu().numpy())
                print("stats:", self.last_clip_debug)

            z_img = self.img_proj(emb)
        else:
            z_img = torch.zeros((vec.shape[0], self.proj_dim), device=self.device, dtype=vec.dtype)
        z = torch.cat([z_img, vec], dim=-1)
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
