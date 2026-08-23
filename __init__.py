WEB_DIRECTORY = "./web"


async def comfy_entrypoint():
    # Keep node/API imports lazy so the format core can be tested and used by tooling
    # without importing a full ComfyUI runtime.
    try:
        from .mmh3_media.nodes import MMH3Extension
    except ImportError:
        from mmh3_media.nodes import MMH3Extension
    return MMH3Extension()


__all__ = ["WEB_DIRECTORY"]
