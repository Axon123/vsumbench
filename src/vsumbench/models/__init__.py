"""Model-side utilities; learned models join through the registry.

This package ships no learned models (project scope decision, 2026-10-13):
researchers bring their own method as a ``Model`` implementation. It hosts
the shared step-to-frame score expansion (``score_space``) that step-space
models need to emit frame-level ``Scores``; a no-learning ``uniform``
reference lands with the orchestration milestone for CLI smoke tests.
"""
