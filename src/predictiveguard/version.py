from importlib.metadata import PackageNotFoundError, version


def get_app_version() -> str:
    try:
        return version("predictiveguard-ai")
    except PackageNotFoundError:
        return "0.1.0"
