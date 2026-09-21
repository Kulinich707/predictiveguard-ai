from unittest.mock import patch

from predictiveguard.version import get_app_version


def test_version_from_package_metadata() -> None:
    with patch("predictiveguard.version.version", return_value="1.2.3"):
        assert get_app_version() == "1.2.3"
