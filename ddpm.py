import torch


class DDPM:
    def __init__(
        self,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device="cpu",
    ):
        self.timesteps = timesteps
        self.device = torch.device(device)

        # Original DDPM linear beta schedule.
        self.beta = torch.linspace(
            beta_start, beta_end, timesteps, device=self.device
        )
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x0, t, noise):
        """Sample x_t from q(x_t | x_0)."""
        alpha_bar_t = self.alpha_bar[t][:, None, None, None]
        return (
            alpha_bar_t.sqrt() * x0
            + (1.0 - alpha_bar_t).sqrt() * noise
        )

    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        labels=None,
        guidance_scale=1.0,
        capture_steps=None,
    ):
        """Run the reverse process with optional classifier-free guidance."""
        model.eval()
        x = torch.randn(shape, device=self.device)
        if labels is not None:
            labels = torch.as_tensor(
                labels, device=self.device, dtype=torch.long
            )
            if labels.ndim != 1 or labels.size(0) != shape[0]:
                raise ValueError("labels must have shape [batch_size]")
        unconditional = labels is None or bool((labels == -1).all())
        capture_steps = set(capture_steps or [])
        frames = []

        for frame_index, step in enumerate(reversed(range(self.timesteps))):
            t = torch.full(
                (shape[0],), step, device=self.device, dtype=torch.long
            )
            if unconditional:
                predicted_noise = model(x, t, labels)
            elif guidance_scale == 1.0:
                predicted_noise = model(x, t, labels)
            else:
                conditional_noise = model(x, t, labels)
                unconditional_noise = model(x, t, None)
                predicted_noise = unconditional_noise + guidance_scale * (
                    conditional_noise - unconditional_noise
                )

            alpha_t = self.alpha[step]
            alpha_bar_t = self.alpha_bar[step]
            beta_t = self.beta[step]

            model_mean = (
                x - beta_t / torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
            ) / torch.sqrt(alpha_t)

            if step > 0:
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(beta_t) * noise
            else:
                x = model_mean

            if frame_index in capture_steps:
                frames.append(x.clamp(-1, 1).cpu())

        images = x.clamp(-1, 1)
        if capture_steps:
            return images, torch.stack(frames)
        return images
