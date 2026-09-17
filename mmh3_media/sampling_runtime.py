"""Runtime sampler catalogs and H3 scheduler dispatch, including beta57."""
from .errors import MMH3ResourceError

DEFAULT_SAMPLERS = ['res_multistep', 'euler', 'er_sde', 'heun', 'dpmpp_2m', 'dpmpp_2m_sde']
DEFAULT_SCHEDULERS = ['simple', 'normal', 'beta', 'beta57', 'karras', 'exponential', 'sgm_uniform', 'ddim_uniform', 'linear_quadratic', 'kl_optimal']


def sampler_options():
    import comfy.samplers
    return list(dict.fromkeys(['res_multistep', *comfy.samplers.KSampler.SAMPLERS]))


def scheduler_options():
    import comfy.samplers
    return list(dict.fromkeys([*comfy.samplers.SCHEDULER_NAMES, 'beta57']))


def calculate_sigmas(model_sampling, scheduler, steps):
    import comfy.samplers
    if scheduler == 'beta57':
        return comfy.samplers.beta_scheduler(model_sampling, steps, alpha=0.5, beta=0.7)
    if scheduler not in comfy.samplers.SCHEDULER_NAMES:
        raise MMH3ResourceError(f'Scheduler {scheduler!r} is not installed in ComfyUI')
    return comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)


def scheduler_sigmas(model, scheduler, steps, denoise=1.0):
    import torch
    if steps < 1 or not 0 <= denoise <= 1:
        raise MMH3ResourceError('Scheduler requires positive steps and denoise in 0–1')
    if denoise == 0:
        return torch.empty(0, dtype=torch.float32)
    total = int(steps / denoise) if denoise < 1 else steps
    return calculate_sigmas(model.get_model_object('model_sampling'), scheduler, total).cpu()[-(steps + 1):]
