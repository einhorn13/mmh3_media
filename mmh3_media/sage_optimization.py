"""Compose a KJ dense backend with an existing sparse/architecture hook."""
from .errors import MMH3ResourceError
from .optimization_contract import optimization_state, STATE_KEY


def validate_sage_application(model):
    state = optimization_state(model)
    if state.get("sage_attention") or state["attention"] == "sage_attention_kj":
        raise MMH3ResourceError("MODEL already has SageAttention; duplicate application refused")


def apply_sage_attention(model, node_class, mode, allow_compile=False, preserve_strategy=True):
    validate_sage_application(model)
    previous = model.model_options.get("transformer_options", {}).get("optimized_attention_override")
    patched = node_class().patch(model, mode, allow_compile=allow_compile)[0]
    options = patched.model_options.setdefault("transformer_options", {})
    sage = options.get("optimized_attention_override")
    if not callable(sage) or sage is previous:
        raise MMH3ResourceError("KJNodes did not install the selected SageAttention backend")

    if previous is not None and preserve_strategy:
        def composed(func, *args, **kwargs):
            # The strategy still decides which calls are sparse. Only its dense
            # fall-through receives Sage, without invoking Comfy's hook twice.
            def dense(*dense_args, **dense_kwargs):
                return sage(func, *dense_args, **dense_kwargs)
            return previous(dense, *args, **kwargs)
        options["optimized_attention_override"] = composed

    state = dict(optimization_state(patched))
    state["sage_attention"] = {"mode": mode, "allow_compile": bool(allow_compile)}
    patched.model_options[STATE_KEY] = state
    return patched
