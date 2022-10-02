from typing import Tuple
import json

import jax
import jax.numpy as jnp
from jax.experimental import maps
from jax.experimental import PartitionSpec
from jax.experimental.pjit import pjit
from flax.traverse_util import flatten_dict
from flax.core.frozen_dict import freeze, unfreeze

from .pjit_partition import set_partitions
from optax import Params
import numpy as np

from transformers import FlaxOPTForCausalLM, AutoTokenizer

Array = jnp.ndarray

devices = np.asarray(jax.devices())

num_mp_devices = 8
num_dp_devices = int(jax.device_count() / num_mp_devices)

devices_logical_mesh = devices.reshape((num_dp_devices, num_mp_devices))

# mp: model/tensor parallelism
# dp: data parallelism
mesh_axis_names = ("dp", "mp")
print("Logical mesh:", devices_logical_mesh)

mesh = maps.Mesh(devices_logical_mesh, mesh_axis_names)

base_model_name = "facebook/opt-13b"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
text_prompt = tokenizer(["McGill is located"] * num_dp_devices)  # type: ignore
text_input_ids: Array = jnp.asarray(text_prompt.input_ids)  # type: ignore


model: FlaxOPTForCausalLM
params: Params
model, params = FlaxOPTForCausalLM.from_pretrained(  # type: ignore
    base_model_name, dtype=jnp.float16, _do_init=False  # type: ignore
)

param_shape = jax.tree_util.tree_map(jnp.shape, params)
with open("params.json", "w") as param_shape_json_file:
    json.dump(param_shape, param_shape_json_file, indent=2)

with open("param_shape.json", "w") as param_shape_json_file:
    flatten_dict_output = {
        str(k): jnp.shape(v) for k, v in flatten_dict(params).items()  # type: ignore
    }
    json.dump(flatten_dict_output, param_shape_json_file, indent=2)

data_partition_specs = PartitionSpec()
extra_param_keys = list(model._missing_keys)
initial_partition_specs = set_partitions(params)
filled_param_partition_specs = set_partitions(params, extra_keys=extra_param_keys)


with open("param_partition.json", "w") as param_partition_json_file:
    json.dump(unfreeze(initial_partition_specs), param_partition_json_file, indent=2)

prng_key = jax.random.PRNGKey(0)


def generate_fn_(input_ids: Array, params: Params):
    return model.generate(
        input_ids,
        params=params,  # type: ignore
        max_length=50,
        num_beams=5,
        early_stopping=True,
    )


def init_param(
    prng_key: jax.random.PRNGKeyArray, input_shape: Tuple[int], params: Params
) -> Params:
    """
    Fill in (initialize) missing model params.
    See https://github.com/huggingface/transformers/issues/15766
    """
    ...


init_param = pjit(
    model.init_weights,
    static_argnums=(1,),
    in_axis_resources=(PartitionSpec(), initial_partition_specs),  # type: ignore
    out_axis_resources=filled_param_partition_specs,  # type: ignore
)


pjit_generate_function = pjit(
    generate_fn_,
    in_axis_resources=(PartitionSpec(), filled_param_partition_specs),  # type: ignore
    out_axis_resources=PartitionSpec(),  # type: ignore
)

with maps.Mesh(devices_logical_mesh, mesh.axis_names):
    updated_params = init_param(
        jax.random.PRNGKey(0),
        text_input_ids.shape,
        freeze(params),  # type: ignore
    )
    beam_results = pjit_generate_function(text_input_ids, updated_params)

print(beam_results)

text_output = tokenizer.batch_decode(
    beam_results.sequences,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]
print(text_output.splitlines())
