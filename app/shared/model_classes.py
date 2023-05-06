from transformers import GPT2Tokenizer, GPT2LMHeadModel, LlamaForCausalLM, LlamaTokenizer, BloomForCausalLM, BloomTokenizerFast

MODEL_CLASSES = {
        "gpt2": (GPT2LMHeadModel, GPT2Tokenizer, "gpt2"),
        "llama": (LlamaForCausalLM, LlamaTokenizer, "decapoda-research/llama-7b-hf"),
        "bloom": (BloomForCausalLM, BloomTokenizerFast, "bigscience/bloom-560m"),
        "alpaca": (LlamaForCausalLM, LlamaTokenizer, "chavinlo/alpaca-native")
    }