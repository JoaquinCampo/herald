# numpy & torch idioms

Code-quality rubric entries for **numpy & torch idioms** (7 entries). See `README.md` for the full topic index.

## Thread one Generator through call sites; never reseed global np.random in a loop  ·  `high`

Create one `rng = np.random.default_rng(seed)` and pass it down as an explicit parameter (`def f(..., rng): rng = np.random.default_rng(rng)`). Do not call `np.random.seed()` / `np.random.shuffle()` / `np.random.choice()` on the implicit global state, and never call `default_rng(seed)` inside a loop (that resets the stream every iteration). For worker processes or parallel sampling, derive independent streams with `rng.spawn(n)` rather than seeding each worker with `seed + i`.


**Why:** The implicit global RandomState is shared mutable state: a reseed anywhere in the codebase (or a library) silently changes your draws, and it is not thread/process-safe. Reseeding inside a loop produces correlated or identical draws; `seed + i` per worker gives statistically correlated streams. spawn() guarantees independence. Ruff NPY002 would flag the global calls but is not in this repo's ruleset, and it cannot see the loop-reseed or the correlated-worker-seed judgment errors at all.


**Avoid:**
```python
np.random.seed(0)
for batch in batches:
    rng = np.random.default_rng(0)   # resets stream every iter
    idx = np.random.choice(n, k)     # implicit global state
```

**Prefer:**
```python
def sample(batches, k, rng):
    rng = np.random.default_rng(rng)
    for batch in batches:
        idx = rng.choice(n, k, replace=False)
```

Source: [Scientific Python — Best Practices for Using NumPy's Random Number Generators](https://blog.scientific-python.org/numpy/numpy-rng/)

## Know view vs copy: fancy indexing copies, slicing aliases the base array  ·  `high`

Treat basic slicing (`a[1:3]`, `a[:, 0]`) as a view that aliases the original buffer: writing to it mutates the base array. Advanced/fancy indexing (`a[[1,2]]`, boolean masks) always returns a copy, so in-place edits to the result are silently dropped from the source. When you need an independent array, call `.copy()` explicitly; when you need an in-place edit to stick, index in place rather than editing a fancy-indexed copy.


**Why:** These two index forms look almost identical but have opposite aliasing semantics. A slice you mutate can corrupt data elsewhere that shares the buffer; a fancy-indexed result you mutate looks like it worked but never touches the source. Both are classic silent-correctness bugs no linter detects. `ravel()`/`reshape()` may return a view or a copy depending on contiguity, compounding the ambiguity.


**Avoid:**
```python
rows = embeddings[[0, 5, 9]]   # fancy index -> COPY
rows[:] = normalize(rows)       # edit lost; embeddings unchanged
view = embeddings[0:3]
view *= 2                       # silently scales embeddings[0:3] too
```

**Prefer:**
```python
idx = [0, 5, 9]
embeddings[idx] = normalize(embeddings[idx])  # write back through index
chunk = embeddings[0:3].copy()                # explicit independent copy
```

Source: [NumPy v2 Manual — Copies and views](https://numpy.org/doc/stable/user/basics.copies.html)

## Prefer torch.inference_mode() over torch.no_grad() for pure inference  ·  `medium`

Wrap inference/encoding forward passes in `torch.inference_mode()` rather than `torch.no_grad()`. Only fall back to `no_grad()` if a tensor produced inside the block must later participate in autograd, because tensors created under inference_mode are tagged and cannot be used in any later grad-tracking computation.


**Why:** inference_mode is the stronger guarantee: beyond skipping gradient recording (like no_grad) it also drops view tracking and version-counter bookkeeping, so it runs faster and uses less memory. Reaching for no_grad out of habit leaves that speedup on the table. The trade-off (tagged tensors) only bites if you reuse outputs in a graph, which encoder/eval code never does.


**Avoid:**
```python
with torch.no_grad():
    emb = model(input_ids)  # inference-only encoder pass
```

**Prefer:**
```python
with torch.inference_mode():
    emb = model(input_ids)  # faster: also disables view/version tracking
```

Source: [Zach Mueller — Inference in PyTorch: what do the wrappers mean? What's best?](https://muellerzr.github.io/blog/PyTorchInference.html)

## np.vectorize is not vectorization; rewrite with native ufuncs/broadcasting  ·  `medium`

Do not reach for `np.vectorize` (or `np.apply_along_axis`) expecting a speedup over a Python loop. Express row/element logic with native array operations: broadcasting, ufuncs, boolean masks, `np.einsum`, matrix ops. Reserve `np.vectorize` for convenience on genuinely scalar-only functions where performance does not matter.


**Why:** The NumPy docs state plainly that vectorize 'is provided primarily for convenience, not for performance' and that 'the implementation is essentially a for loop.' Code that wraps a scalar function in np.vectorize reads as if it were optimized but still pays full Python per-element overhead. Real speedups come only from pushing the loop into C via native ops.


**Avoid:**
```python
score = np.vectorize(lambda q, d: cosine(q, d))(queries, docs)
```

**Prefer:**
```python
q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
d = docs / np.linalg.norm(docs, axis=1, keepdims=True)
score = q @ d.T   # broadcast + BLAS, no per-row Python
```

Source: [NumPy v2 Manual — numpy.vectorize](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html)

## Create tensors directly on the target device and keep dtype at float32  ·  `medium`

Construct tensors with `device=` (and `dtype=`) at creation rather than building on CPU then `.to(device)`. When bringing NumPy data into torch, cast to float32 explicitly: NumPy defaults to float64, and a float64 tensor on GPU is far slower and uses double the memory. Avoid silently mixing float32 and float64, which upcasts the whole op to float64.


**Why:** The tuning guide says to 'produce the output directly on the target device' instead of `torch.rand(size).cuda()`, avoiding an extra host->device copy. The PyTorch NumPy-compile post notes consumer GPUs are 'rather sluggish when running operations on float64'; switching generation to float32 made the CUDA path 40% faster. NumPy arrays default to float64, so `torch.from_numpy(np_arr)` silently carries float64 onto the GPU unless you cast.


**Avoid:**
```python
buf = torch.zeros(n, d)            # CPU float32
buf = buf.to(device)
x = torch.from_numpy(np_feats)     # inherits float64 from numpy
```

**Prefer:**
```python
buf = torch.zeros(n, d, device=device)
x = torch.from_numpy(np_feats.astype(np.float32, copy=False)).to(device)
```

Source: [PyTorch Blog — Compiling NumPy code into C++ or CUDA via torch.compile](https://pytorch.org/blog/compiling-numpy-code/)

## Use detach().cpu().numpy() in that order; force=True is the documented shortcut  ·  `medium`

To convert a tensor that may require grad and/or live on GPU into a NumPy array, call `t.detach().cpu().numpy()` (detach first, then move to CPU, then convert). Plain `t.numpy()` raises if the tensor requires grad, is on CUDA, or has a conjugate bit. If you want one call, `t.numpy(force=True)` is documented as equivalent to `t.detach().cpu().resolve_conj().resolve_neg().numpy()`.


**Why:** `Tensor.numpy()` with the default `force=False` requires the tensor to be on CPU, not require grad, and have no conjugate bit set, so the explicit chain is the safe form. The PyTorch docs define `force=True` as exactly that detach/cpu/resolve chain, which is the authoritative ordering. Getting the order wrong (e.g. `.cpu().numpy()` on a grad-tracking tensor) raises at runtime, not at lint time.


**Avoid:**
```python
arr = logits.numpy()          # raises if on CUDA or requires_grad
```

**Prefer:**
```python
arr = logits.detach().cpu().numpy()   # or: logits.numpy(force=True)
```

Source: [PyTorch 2.12 docs — torch.Tensor.numpy](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.numpy.html)

## Remember torch.from_numpy() and tensor.numpy() share memory with the source  ·  `medium`

Treat `torch.from_numpy(arr)` and a CPU `tensor.numpy()` as aliases of the same buffer: the docs state the tensor and ndarray 'share the same memory' and modifications to one are reflected in the other. If you will mutate one side (in-place normalization, scaling, augmentation) and need the other to stay intact, copy explicitly (`arr.copy()`, `torch.tensor(arr)`, or `.clone()`).


**Why:** The shared-buffer behavior is documented and deliberate, but it produces action-at-a-distance bugs: an in-place op on the tensor silently rewrites the NumPy array a caller still holds (and vice versa). `torch.tensor(arr)` copies by default and is the safe choice when independence matters. A linter cannot tell an intentional alias from an accidental one.


**Avoid:**
```python
t = torch.from_numpy(features)   # shares buffer
t.mul_(scale)                    # silently scales `features` too
```

**Prefer:**
```python
t = torch.from_numpy(features).clone()   # independent
# or torch.tensor(features) which copies by default
t.mul_(scale)
```

Source: [PyTorch 2.12 docs — torch.from_numpy](https://docs.pytorch.org/docs/2.12/generated/torch.from_numpy.html)
