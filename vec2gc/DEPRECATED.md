This folder's contents have been moved to `src/vec2gc/` to use the recommended src/ package layout.

Please remove this folder once you confirm the `src/vec2gc/` copy is working and your packaging metadata uses `where = "src"` in `pyproject.toml`.

Why: src-layout prevents accidental imports from the project root when the package is installed, and is the modern recommended layout for Python projects.
