from diffusers import FlaxStableDiffusionPipeline
import numpy as np
import jax

from jax.experimental import maps
from jax.experimental import PartitionSpec
from jax.experimental.pjit import pjit

import wandb
import datetime

prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(  # type: ignore
    "fusing/stable-diffusion-flax-new", use_auth_token=True
)
del params["safety_checker"]

# pjit and mesh
devices = np.asarray(jax.devices())
devices_mesh_reshaped = devices.reshape((-1, 1))
mesh_axis_names = ("program", "data")

mesh = maps.Mesh(devices_mesh_reshaped, mesh_axis_names)

in_axis_resources = PartitionSpec()
out_axis_resource = PartitionSpec()
# p_sample = pmap(pipeline.__call__, static_broadcasted_argnums=(3,))
p_sample = pjit(
    pipeline.__call__,
    in_axis_resources=in_axis_resources,  # type: ignore
    out_axis_resources=out_axis_resource,  # type: ignore
    static_argnums=(3,),
)

# prep prompts
while True:
    prompt = input("Enter prompt: ")
    run = wandb.init(
        project="pjit-image-generation",
        entity="jacobthebanana",
        name=datetime.datetime.now().isoformat(),
    )
    wandb.run.name = datetime.datetime.now().isoformat()
    start_time = datetime.datetime.now()

    num_samples = 1
    prompt_ids = pipeline.prepare_inputs([prompt])

    # run
    with maps.Mesh(devices_mesh_reshaped, mesh_axis_names):
        images = p_sample(prompt_ids, params, prng_seed, num_inference_steps).images

    # get pil images
    images_pil = pipeline.numpy_to_pil(
        np.asarray(images.reshape((num_samples,) + images.shape[-3:]))
    )

    print("Images should be good")
    time_taken = datetime.datetime.now() - start_time
    wandb.log(
        {
            "prompt": prompt,
            "media": wandb.Image(images_pil[0]),
            "time taken": str(time_taken),
        }
    )
    run.finish()
