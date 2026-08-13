# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import open_clip


# class ClipVisionEncoder(nn.Module):
#     """
#     Frozen CLIP vision encoder that maps image -> embedding.
#     Input: torch float32 tensor (B, 3, H, W) in [0,1] or [0,255]
#     Output: (B, D) float32
#     """
#     def __init__(self, model_name="ViT-B-32", pretrained="openai", device="cuda"):
#         super().__init__()
#         self.device = torch.device(device)

#         # ✅ Use from_pretrained to ensure model config matches checkpoint
#         # This typically resolves the QuickGELU mismatch warning.
#         model, _ = open_clip.create_model_from_pretrained(
#             model_name,
#             pretrained=pretrained,
#             device=str(self.device),
#         )
#         self.model = model.eval()
#         for p in self.model.parameters():
#             p.requires_grad = False

#         self.image_size = 224
#         self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
#         self.register_buffer("std",  torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

#         with torch.no_grad():
#             dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
#             emb = self.model.encode_image(dummy)
#         self.embed_dim = emb.shape[-1]

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x.to(self.device)

#         # Accept [0,255] or [0,1]
#         if x.max() > 1.5:
#             x = x / 255.0

#         # Resize to CLIP expected size
#         if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
#             x = F.interpolate(x, size=(self.image_size, self.image_size),
#                               mode="bilinear", align_corners=False)

#         # Normalize (CLIP mean/std)
#         x = (x - self.mean) / self.std

#         emb = self.model.encode_image(x).float()
#         emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
#         return emb
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class ClipVisionEncoder(nn.Module):
    """
    CLIP vision encoder with:
    1) global image embedding
    2) patch token extraction
    3) text encoding
    4) text-guided patch pooling

    Input image: (B, 3, H, W), float32, range [0,1] or [0,255]
    """

    def __init__(
        self,
        model_name="ViT-B-32",
        pretrained="openai",
        device="cuda",
        freeze=True,
        image_size=224,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.image_size = image_size
        self.freeze = freeze

        # create model
        model, _ = open_clip.create_model_from_pretrained(
            model_name,
            pretrained=pretrained,
            device=str(self.device),
        )
        self.model = model.eval()

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
            emb = self.model.encode_image(dummy)
        self.embed_dim = emb.shape[-1]

        # tokenizer
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device).float()

        # accept [0,255] or [0,1]
        if x.max() > 1.5:
            x = x / 255.0

        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        x = (x - self.mean) / self.std
        return x

    def encode_text(self, prompts):
        """
        prompts: list[str]
        return: (N, D), normalized
        """
        tokens = self.tokenizer(prompts).to(self.device)
        if self.freeze:
            with torch.no_grad():
                text_feat = self.model.encode_text(tokens).float()
        else:
            text_feat = self.model.encode_text(tokens).float()

        text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-8)
        return text_feat

    def encode_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: raw image tensor (B,3,H,W)
        return: patch tokens (B, N, D), normalized
        only supports ViT-style CLIP visual backbone
        """
        x = self.preprocess(x)

        visual = self.model.visual

        # ViT patch embedding
        x = visual.conv1(x)                      # (B, C, Gh, Gw)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, C, N)
        x = x.permute(0, 2, 1)                  # (B, N, C)

        class_emb = visual.class_embedding.to(x.dtype)
        cls_tokens = class_emb + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_tokens, x], dim=1)   # (B, N+1, C)

        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.patch_dropout(x)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)                  # (L, B, C)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)                  # (B, L, C)

        x = visual.ln_post(x)

        if visual.proj is not None:
            x = x @ visual.proj                 # (B, L, D)

        patch_tokens = x[:, 1:, :]              # remove cls token -> (B, N, D)
        patch_tokens = patch_tokens / (patch_tokens.norm(dim=-1, keepdim=True) + 1e-8)
        return patch_tokens.float()

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        """
        global image embedding: (B, D)
        """
        x = self.preprocess(x)

        if self.freeze:
            with torch.no_grad():
                emb = self.model.encode_image(x).float()
        else:
            emb = self.model.encode_image(x).float()

        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        return emb

    # def forward_text_guided(
    #     self,
    #     x: torch.Tensor,
    #     fg_prompts,
    #     bg_prompts,
    #     text_guidance_lambda=1.0,
    #     temperature=0.2,
    # ):
    #     """
    #     return:
    #         pooled_feat: (B, D)
    #         weights: (B, N)
    #         patch_tokens: (B, N, D)
    #         fg_scores: (B, N)
    #         bg_scores: (B, N)
    #     """
    #     patch_tokens = self.encode_patch_tokens(x)   # (B, N, D)

    #     fg_text = self.encode_text(fg_prompts).mean(dim=0, keepdim=True)  # (1, D)
    #     bg_text = self.encode_text(bg_prompts).mean(dim=0, keepdim=True)  # (1, D)

    #     fg_text = fg_text.expand(patch_tokens.shape[0], -1)  # (B, D)
    #     bg_text = bg_text.expand(patch_tokens.shape[0], -1)  # (B, D)

    #     fg_scores = torch.sum(patch_tokens * fg_text.unsqueeze(1), dim=-1)  # (B, N)
    #     bg_scores = torch.sum(patch_tokens * bg_text.unsqueeze(1), dim=-1)  # (B, N)

    #     # ===== 原始前景-背景分数 =====
    #     scores = fg_scores - text_guidance_lambda * bg_scores  # (B, N)

    #     # ===== 关键增强1：标准化，放大小差异 =====
    #     scores = (scores - scores.mean(dim=-1, keepdim=True)) / (
    #         scores.std(dim=-1, keepdim=True) + 1e-6
    #     )

    #     # ===== 关键增强2：temperature softmax，让权重更尖锐 =====
    #     weights = torch.softmax(scores / temperature, dim=-1)  # (B, N)

    #     pooled_feat = torch.sum(patch_tokens * weights.unsqueeze(-1), dim=1)  # (B, D)
    #     pooled_feat = pooled_feat / (pooled_feat.norm(dim=-1, keepdim=True) + 1e-8)

    #     return (
    #         pooled_feat.float(),
    #         weights.float(),
    #         patch_tokens.float(),
    #         fg_scores.float(),
    #         bg_scores.float(),
    #     )

    def forward_text_guided(
        self,
        x: torch.Tensor,
        fg_prompts,
        bg_prompts,
        text_guidance_lambda=1.0,
        temperature=0.2,
        use_spatial_prior=True,
    ):
        """
        return:
            pooled_feat: (B, D)
            weights: (B, N)
            patch_tokens: (B, N, D)
            fg_scores: (B, N)
            bg_scores: (B, N)
        """
        patch_tokens = self.encode_patch_tokens(x)   # (B, N, D)

        fg_text = self.encode_text(fg_prompts).mean(dim=0, keepdim=True)  # (1, D)
        bg_text = self.encode_text(bg_prompts).mean(dim=0, keepdim=True)  # (1, D)

        fg_text = fg_text.expand(patch_tokens.shape[0], -1)  # (B, D)
        bg_text = bg_text.expand(patch_tokens.shape[0], -1)  # (B, D)

        fg_scores = torch.sum(patch_tokens * fg_text.unsqueeze(1), dim=-1)  # (B, N)
        bg_scores = torch.sum(patch_tokens * bg_text.unsqueeze(1), dim=-1)  # (B, N)

        # foreground - background
        scores = fg_scores - text_guidance_lambda * bg_scores  # (B, N)

        # standardize scores to amplify small differences
        scores = (scores - scores.mean(dim=-1, keepdim=True)) / (
            scores.std(dim=-1, keepdim=True) + 1e-6
        )

        # sharpen attention
        weights = torch.softmax(scores / temperature, dim=-1)  # (B, N)

        # optional spatial prior: suppress top rows (sky / horizon)
        if use_spatial_prior:
            num_patches = weights.shape[1]
            grid_size = int(num_patches ** 0.5)

            if grid_size * grid_size == num_patches:
                spatial_mask = torch.ones(
                    (grid_size, grid_size),
                    device=weights.device,
                    dtype=weights.dtype,
                )

                # top rows are usually sky / horizon noise
                spatial_mask[0, :] = 0.1
                if grid_size > 1:
                    spatial_mask[1, :] = 0.4

                # bottom row may contain ship body, keep but slightly suppress
                spatial_mask[-1, :] = 0.7

                spatial_mask = spatial_mask.reshape(1, -1)  # (1, N)
                weights = weights * spatial_mask
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        pooled_feat = torch.sum(patch_tokens * weights.unsqueeze(-1), dim=1)  # (B, D)
        pooled_feat = pooled_feat / (pooled_feat.norm(dim=-1, keepdim=True) + 1e-8)

        return (
            pooled_feat.float(),
            weights.float(),
            patch_tokens.float(),
            fg_scores.float(),
            bg_scores.float(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        default behavior: keep backward compatibility with old code
        returns global image embedding (B, D)
        """
        return self.forward_global(x)