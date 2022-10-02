import jax

from jax.experimental import maps
from jax.experimental import PartitionSpec
from jax.experimental.pjit import pjit
import jax.numpy as jnp
import numpy as np

devices = np.asarray(jax.devices())
devices_logical_mesh = devices.reshape((4, 2))
print("Logical mesh:", devices_logical_mesh)

mesh = maps.Mesh(devices_logical_mesh, ("program", "data"))

input_data = np.arange(8 * 2).reshape((8, 2))
print("Input data shape:", input_data.shape)

in_axis_resources = PartitionSpec()
out_axis_resources = PartitionSpec("data", None)


def example_function_(x, y):
    return x


example_function_pjit = pjit(
    example_function_,
    in_axis_resources=(PartitionSpec("program"), PartitionSpec()),  # type: ignore
    out_axis_resources=out_axis_resources,  # type: ignore
)

print(example_function_pjit)
print("Mesh devices:", mesh.devices)
print("Mesh axes:", mesh.axis_names)
print("Output partition specs:", out_axis_resources)

with maps.Mesh(mesh.devices, mesh.axis_names):  # type: ignore
    output_data = example_function_pjit(input_data, jnp.ones(1))

print("Output shape:", output_data.shape)
print(output_data)

print("Device buffers:")
print(output_data.device_buffers)
