from importlib.metadata import version


def get_app_version() -> str:
    return version("predictiveguard-ai")
