from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package isn’t installed (e.g. dev checkout); fall back to a sensible default
    __version__ = "0.0.0"
