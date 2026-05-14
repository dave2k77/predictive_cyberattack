from predictive_cyberattack.environment import _parse_java_major_version


def test_parse_modern_java_version():
    assert _parse_java_major_version('openjdk version "17.0.18" 2026-01-20') == 17


def test_parse_legacy_java_version_format():
    assert _parse_java_major_version('java version "1.8.0_402"') == 8
