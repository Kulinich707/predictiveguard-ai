from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

from predictiveguard.version import get_app_version


def test_version_from_package_metadata() -> None:
    with patch("predictiveguard.version.version", return_value="1.2.3"):
        assert get_app_version() == "1.2.3"


def test_version_fallback() -> None:
    with patch("predictiveguard.version.version", side_effect=PackageNotFoundError):
        assert get_app_version() == "0.1.0"
