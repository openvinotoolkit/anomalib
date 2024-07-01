AVAILABLE_MODELS = {
    "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "llava_gemma": "LlavaGemmaForCausalLM, LlavaGemmaConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    # "llava_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",
    "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
    # Add other models as needed
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from anomalib.models.image.llava_next.model.language_model.{model_name} import {model_classes}")
    except ImportError:
        # import traceback
        # traceback.print_exc()
        print(f"Failed to import {model_name} from llava.language_model.{model_name}")
