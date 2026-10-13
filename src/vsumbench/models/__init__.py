"""Model-side utilities and the built-in reference model.

This package ships no learned models (project scope decision, 2026-10-13):
researchers bring their own method as a ``Model`` implementation. It hosts
the shared step-to-frame score expansion (``score_space``) that step-space
models need to emit frame-level ``Scores``, and the learning-free
``uniform`` reference for CLI smoke tests and examples.
"""
