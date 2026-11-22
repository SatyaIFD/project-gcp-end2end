"""Initial package smoke tests verifying installation and version metadata."""

from claims_triage import __version__


def test_package_version_is_semver() -> None:
    parts = __version__.split(".")
    expected_semver_parts = 3
    assert len(parts) == expected_semver_parts
    assert all(part.isdigit() for part in parts)
