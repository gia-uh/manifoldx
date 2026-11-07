"""three.js is vendored in the wheel, pinned, and makes no request outside its folder."""

import re

from manifoldx.web.export import THREE_VERSION, VENDOR

FILES = [
    "three.module.js",
    "three.core.js",
    "addons/loaders/GLTFLoader.js",
    "addons/utils/BufferGeometryUtils.js",
    "addons/utils/SkeletonUtils.js",
    "LICENSE",
    "VERSION",
    "SOURCE",
]
# An import or re-export statement starts a line (the addons' JSDoc quotes import lines after
# " * "); its specifier ends it, maybe lines later. Bare and dynamic imports count too.
SPECIFIER = re.compile(
    r"""^(?:import|export)\b[^;'"]*?\bfrom\s*["']([^"']+)["']|\bimport\s*\(?\s*["']([^"']+)["']""",
    re.M,
)


def test_every_vendored_file_is_there():
    assert THREE_VERSION == "0.186.1"
    missing = [f for f in FILES if not (VENDOR / f).is_file()]
    assert not missing


def test_version_is_the_pinned_one():
    assert (VENDOR / "VERSION").read_text().strip() == THREE_VERSION
    assert f"three-{THREE_VERSION}.tgz" in (VENDOR / "SOURCE").read_text()


def test_imports_stay_inside_the_folder():
    vendored = {(VENDOR / f).resolve() for f in FILES}
    for f in FILES:
        if not f.endswith(".js"):
            continue
        path = VENDOR / f
        specifiers = [a or b for a, b in SPECIFIER.findall(path.read_text())]
        assert specifiers or f == "three.core.js", f"{f} imports nothing"
        for s in specifiers:
            if s == "three":
                continue
            assert s.startswith("."), f"{f} imports {s}"
            assert (path.parent / s).resolve() in vendored, f"{f} imports {s}, not vendored"


def test_the_module_keeps_its_license_header():
    head = (VENDOR / "three.module.js").read_bytes()[:300].decode("utf8", "replace")
    assert "@license" in head
    assert "MIT" in head
