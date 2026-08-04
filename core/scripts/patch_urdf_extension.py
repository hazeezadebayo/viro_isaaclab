#!/usr/bin/env python3
"""Idempotently apply a headless-safety patch to Isaac Sim's URDF importer extension.

The stock extension creates a Qt window and an import delegate in ``Extension.on_startup``.
Under a headless container (no display / no Vulkan) this never completes, so Isaac Sim hangs
while loading any task that uses ``UrdfFileCfg`` (e.g. the AMR task).

This script adds a ``_is_headless()`` guard (using the ``/app/window/hideUi`` carb setting) so the
window and delegate are skipped when running headless. It is idempotent: if the marker function
already exists, it does nothing. It can be run from the host against a running container or inside
the container.

Usage (from host, container must be running):
  python core/scripts/patch_urdf_extension.py --container isaac-rl-studio

Or (inside container, extension already installed):
  python core/scripts/patch_urdf_extension.py --in-container

The patch keeps the non-headless path fully intact, so a GUI session is unaffected.
"""

import argparse
import re
import subprocess
import sys
import tempfile

_MARKER = "def _is_headless():"

_PATCH_ON_STARTUP_OLD = """    def on_startup(self, ext_id):
        self._ext_id = ext_id
        self.window = None
        self._delegate = None

        self.window = UrdfImporter(ext_id)

        self._delegate = UrdfImporterDelegate(
            "Urdf Importer", ["(.*\\\\.urdf$)|(.*\\\\.URDF$)"], ["Urdf Files (*.urdf, *.URDF)"], ext_id
        )
        ai.register_importer(self._delegate)
"""

_PATCH_ON_STARTUP_NEW = """    def on_startup(self, ext_id):
        self._ext_id = ext_id
        self.window = None
        self._delegate = None
        self._is_headless = _is_headless()

        if self._is_headless:
            return

        self.window = UrdfImporter(ext_id)

        self._delegate = UrdfImporterDelegate(
            "Urdf Importer", ["(.*\\\\.urdf$)|(.*\\\\.URDF$)"], ["Urdf Files (*.urdf, *.URDF)"], ext_id
        )
        ai.register_importer(self._delegate)
"""

_PATCH_HELPER_FUNC = """
def _is_headless():
    try:
        return bool(carb.settings.get_settings().get("/app/window/hideUi"))
    except Exception:
        return False


"""


def patch_content(src: str) -> str:
    if _MARKER in src:
        return src
    if _PATCH_ON_STARTUP_OLD not in src:
        raise RuntimeError("Could not locate the URDF importer on_startup body; patch aborted.")
    src = src.replace(_PATCH_ON_STARTUP_OLD, _PATCH_ON_STARTUP_NEW)
    # insert the helper right before `class Extension(omni.ext.IExt):`
    match = re.search(r"^class Extension\(omni\.ext\.IExt\):", src, re.MULTILINE)
    if match is None:
        raise RuntimeError("Could not locate the Extension class; patch aborted.")
    src = src[: match.start()] + _PATCH_HELPER_FUNC + src[match.start() :]
    return src


def patch_via_container(container: str, ext_path: str) -> None:
    """Fetch extension.py from the container, patch locally, push it back, clear pycache."""
    dst = f"{container}:{ext_path}"
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8")
    tmp_path = tmp.name
    tmp.close()

    subprocess.run(["docker", "cp", dst, tmp_path], check=True)
    with open(tmp_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    patched = patch_content(src)
    with open(tmp_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(patched)
    subprocess.run(["docker", "cp", tmp_path, dst], check=True)

    ext_dir = ext_path.rsplit("/", 1)[0]
    subprocess.run(
        ["docker", "exec", container, "bash", "-lc", f"find {ext_dir} -name '__pycache__' -type d -exec rm -rf {{}} + 2>/dev/null"],
        check=True,
    )
    print(f"[OK] Patched and synced {dst}")


def patch_in_container(ext_path: str) -> None:
    with open(ext_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    patched = patch_content(src)
    if patched != src:
        with open(ext_path, "w", encoding="utf-8", newline="") as fh:
            fh.write(patched)
    print(f"[OK] Patched {ext_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", help="Container name to patch via docker cp (host-side).")
    parser.add_argument("--in-container", action="store_true", help="Patch the locally installed extension.py.")
    parser.add_argument(
        "--ext-path",
        default=(
            "/root/.local/share/ov/data/exts/v2/"
            "isaacsim.asset.importer.urdf-2.3.14+106.5.0.lx64.r.cp310/"
            "isaacsim/asset/importer/urdf/scripts/extension.py"
        ),
        help="Path to extension.py inside the container.",
    )
    args = parser.parse_args()

    if not (args.container or args.in_container):
        parser.error("Either --container NAME or --in-container is required.")

    if args.container:
        patch_via_container(args.container, args.ext_path)
    else:
        patch_in_container(args.ext_path)


if __name__ == "__main__":
    main()
