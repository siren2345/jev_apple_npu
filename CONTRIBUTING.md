# Contributing

This project targets macOS and Apple Silicon. Keep pull requests focused and include a typed request example whenever changing the API contract.

Before opening a pull request, run:

```sh
python -m pytest -q
python -m build --wheel --no-isolation
```

Do not commit downloaded Hugging Face weights, generated `.mlpackage` directories, API keys, customer state, or Core ML compilation caches. The repository intentionally regenerates model artifacts locally.

For performance changes, record the Mac model, macOS version, compute-unit setting, input shape, warm-up policy, and median of at least five runs.
