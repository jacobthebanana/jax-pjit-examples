import jax.numpy as jnp
import numpy as np

from transformers import FlaxOPTForCausalLM, AutoTokenizer

Array = jnp.ndarray

base_model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
text_prompt = tokenizer(["McGill is located"])  # type: ignore
text_input_ids: Array = jnp.asarray(text_prompt.input_ids)  # type: ignore

model: FlaxOPTForCausalLM
model = FlaxOPTForCausalLM.from_pretrained(base_model_name)  # type: ignore

beam_results = model.generate(text_input_ids)
print(beam_results)
text_output = tokenizer.batch_decode(
    beam_results.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(text_output)
