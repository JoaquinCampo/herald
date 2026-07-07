# Scientific Python

Use this only when the project already uses scientific Python, data frames,
arrays, tensors, or ML tooling.

## NumPy

- Prefer real vectorization through ufuncs and broadcasting over `np.vectorize`,
  which is mostly a convenience loop.
- Be explicit about views versus copies. Slicing can share memory; fancy indexing
  usually copies.
- Thread random generators through call sites instead of reseeding global random
  state in loops.
- Keep dtypes intentional, especially at serialization or model boundaries.

## pandas

- Prefer vectorized operations and clear joins over row-wise loops.
- Be explicit about index semantics.
- Avoid chained assignment ambiguity.
- Validate schema assumptions at data boundaries.

## torch

- Create tensors on the intended device and dtype when practical.
- Use `torch.inference_mode()` for pure inference unless autograd metadata is
  needed.
- Convert tensors to NumPy deliberately, usually via `detach().cpu().numpy()`.
- Avoid CPU-GPU synchronization in hot loops, such as repeated `.item()` calls,
  unless measuring or logging requires it.
- Manage randomness with explicit generators or documented seeds.

## Sources

- NumPy indexing docs: https://numpy.org/doc/stable/user/basics.indexing.html
- NumPy random generator docs: https://numpy.org/doc/stable/reference/random/generator.html
- pandas indexing docs: https://pandas.pydata.org/docs/user_guide/indexing.html
- PyTorch inference mode docs: https://pytorch.org/docs/stable/generated/torch.inference_mode.html
- PyTorch tensor numpy docs: https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html
