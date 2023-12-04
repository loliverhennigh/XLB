from functools import partial

import jax.numpy as jnp

@partial(jnp.vectorize, signature='(n),()->(n)')
def cross_product(a, b):
  print(a.shape, b.shape)
  return a + b


nr = 64
f = jnp.zeros((nr, nr, nr, 27))
id_ = jnp.zeros((nr, nr, nr))
cross_product(f, id_)
