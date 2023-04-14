from transformers import AutoModelForMaskedLM


pytorch_lm = AutoModelForMaskedLM.from_pretrained('mnaylor/psychbert-cased', from_flax=True)
pytorch_lm.save_pretrained("./")