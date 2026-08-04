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

    def _prepare_labels(self, labels, batch_size):
        if labels is None:
            return None
        labels = torch.as_tensor(
            labels, device=self.device, dtype=torch.long
        )
        if labels.ndim != 1 or labels.size(0) != batch_size:
            raise ValueError("labels must have shape [batch_size]")
        return labels

    def _predict_noise(self, model, x, t, labels, guidance_scale):
        unconditional = labels is None or bool((labels == -1).all())
        if unconditional or guidance_scale == 1.0:
            return model(x, t, labels)
        conditional_noise = model(x, t, labels)
        unconditional_noise = model(x, t, None)
        return unconditional_noise + guidance_scale * (
            conditional_noise - unconditional_noise
        )

    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        num_steps=None,
        labels=None,
        guidance_scale=1.0,
        capture_steps=None,
        initial_noise=None,
    ):
        """Run the reverse process with optional classifier-free guidance."""
        num_steps = self.timesteps if num_steps is None else num_steps
        if not 1 <= num_steps <= self.timesteps:
            raise ValueError(
                f"num_steps must be between 1 and {self.timesteps}"
            )
        model.eval()
        if initial_noise is None:
            x = torch.randn(shape, device=self.device)
        else:
            if tuple(initial_noise.shape) != tuple(shape):
                raise ValueError("initial_noise must match shape")
            x = initial_noise.to(self.device)
        labels = self._prepare_labels(labels, shape[0])
        capture_steps = set(capture_steps or [])
        frames = []

        steps = torch.linspace(
            -1, self.timesteps - 1, num_steps + 1, device=self.device
        )[1:].round().long().unique_consecutive()
        steps = steps.flip(0).tolist()

        for frame_index, step in enumerate(steps):
            previous_step = (
                steps[frame_index + 1]
                if frame_index + 1 < len(steps)
                else -1
            )
            t = torch.full(
                (shape[0],), step, device=self.device, dtype=torch.long
            )
            predicted_noise = self._predict_noise(
                model, x, t, labels, guidance_scale
            )

            alpha_bar_t = self.alpha_bar[step]
            alpha_bar_previous = (
                self.alpha_bar[previous_step]
                if previous_step >= 0
                else torch.ones((), device=self.device)
            )
            # Treat each jump as one step in a respaced diffusion process.
            alpha_t = alpha_bar_t / alpha_bar_previous
            beta_t = 1.0 - alpha_t

            model_mean = (
                x - beta_t / torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
            ) / torch.sqrt(alpha_t)

            if previous_step >= 0:
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

    @torch.no_grad()
    def sample_ddim(
        self,
        model,
        shape,
        num_steps=1000,
        eta=0.0,
        labels=None,
        guidance_scale=1.0,
        capture_steps=None,
        initial_noise=None,
    ):
        """Sample with DDIM, optionally adding stochasticity with ``eta``."""
        if not 1 <= num_steps <= self.timesteps:
            raise ValueError(
                f"num_steps must be between 1 and {self.timesteps}"
            )
        if eta < 0:
            raise ValueError("eta must be non-negative")

        model.eval()
        if initial_noise is None:
            x = torch.randn(shape, device=self.device)
        else:
            if tuple(initial_noise.shape) != tuple(shape):
                raise ValueError("initial_noise must match shape")
            x = initial_noise.to(self.device)
        labels = self._prepare_labels(labels, shape[0])
        capture_steps = set(capture_steps or [])
        frames = []

        # Always begin at the noisiest trained timestep. With the full schedule
        # this visits every timestep; shorter schedules skip evenly between them.
        steps = torch.linspace(
            -1, self.timesteps - 1, num_steps + 1, device=self.device
        )[1:].round().long().unique_consecutive()
        steps = steps.flip(0).tolist()

        for frame_index, step in enumerate(steps):
            previous_step = (
                steps[frame_index + 1]
                if frame_index + 1 < len(steps)
                else -1
            )
            t = torch.full(
                (shape[0],), step, device=self.device, dtype=torch.long
            )
            predicted_noise = self._predict_noise(
                model, x, t, labels, guidance_scale
            )

            alpha_bar_t = self.alpha_bar[step]
            alpha_bar_previous = (
                self.alpha_bar[previous_step]
                if previous_step >= 0
                else torch.ones((), device=self.device)
            )
            predicted_x0 = (
                x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
            ) / torch.sqrt(alpha_bar_t)
            sigma = eta * torch.sqrt(
                (1.0 - alpha_bar_previous) / (1.0 - alpha_bar_t)
                * (1.0 - alpha_bar_t / alpha_bar_previous)
            )
            direction = torch.sqrt(
                torch.clamp(1.0 - alpha_bar_previous - sigma.square(), min=0.0)
            ) * predicted_noise
            x = torch.sqrt(alpha_bar_previous) * predicted_x0 + direction
            if previous_step >= 0 and eta > 0:
                x = x + sigma * torch.randn_like(x)

            if frame_index in capture_steps:
                frames.append(x.clamp(-1, 1).cpu())

        images = x.clamp(-1, 1)
        if capture_steps:
            return images, torch.stack(frames)
        return images
