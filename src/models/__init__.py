from .hf_models import hf_instruct_pred, hf_pred
from .openai_models import gpt_pred, o1_pred
from .togetherAI_models import r1_pred


def get_model(cfg, *args, version=None, **kwargs):
    getter = _get_model_instance(cfg.model.name)
    model = getter(**cfg.model, **kwargs)
    return model


def _get_model_instance(name):

    if "llama" in name:
        if "nstruct" in name:
            model = hf_instruct_pred
        else:
            model = hf_pred
    elif "deepseek" in name:
        model = r1_pred
    elif ("o1" in name) or ("o3" in name):
        model = o1_pred
    elif "gpt" in name:
        model = gpt_pred
    else:
        raise ValueError("No valid model name, got: {:}".format(name))

    return model
