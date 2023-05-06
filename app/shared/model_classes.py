from transformers import GPT2Tokenizer, GPT2LMHeadModel, LlamaForCausalLM, LlamaTokenizer, BloomForCausalLM, BloomTokenizerFast

from petals import DistributedBloomForCausalLM

MODEL_CLASSES = {
        "gpt2": (GPT2LMHeadModel, GPT2Tokenizer, "gpt2"),
        "llama": (LlamaForCausalLM, LlamaTokenizer, "decapoda-research/llama-7b-hf"),
        "bloom_petals": (DistributedBloomForCausalLM, BloomTokenizerFast, "bigscience/bloom-7b1-petals"),
        "bloom": (BloomForCausalLM, BloomTokenizerFast, "bigscience/bloom-560m"),
        "alpaca": (LlamaForCausalLM, LlamaTokenizer, "chavinlo/alpaca-native")
    }