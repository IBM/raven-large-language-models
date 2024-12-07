from .hf_models import hf_instruct_pred, hf_pred
from .openai_models import gpt_pred


def get_model(cfg, *args, version=None, **kwargs):
    getter = _get_model_instance(cfg.model.name)
    model = getter(**cfg.model, **kwargs)
    return model


def _get_model_instance(name):

    if "llama" in name or "granite" in name:
        if "nstruct" in name:
            model = hf_instruct_pred
        else:
            model = hf_pred
    elif "gpt" in name:
        model = gpt_pred
    else:
        raise ValueError("No valid model name, got: {:}".format(name))

    return model


# if model_name[:3] == "opt":
#     torch.cuda.empty_cache()
#     free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
#     max_memory = f"{free_in_GB-2}GB"
#     n_gpus = torch.cuda.device_count()
#     max_memory = {i: max_memory for i in range(n_gpus)}
#     print(max_memory)
#     self.model = AutoModelForCausalLM.from_pretrained(
#         "facebook/" + model_name,
#         device_map="auto",
#         load_in_8bit=True,
#         max_memory=max_memory,
#     )
#     self.tokenizer = AutoTokenizer.from_pretrained(
#         "facebook/" + model_name, use_fast=False
#     )
# elif "bloom" in model_name:
#     self.model = pipeline(
#         "text-generation", model=model_name, device_map="auto"
#     )  # "auto")
