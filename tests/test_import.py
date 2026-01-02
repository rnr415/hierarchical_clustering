def test_pkg_importable():
    """Simple smoke test: package imports without optional heavy deps installed."""
    import vec2gc

    # Ensure version attribute exists
    assert hasattr(vec2gc, "__version__")
    assert isinstance(vec2gc.__version__, str)
