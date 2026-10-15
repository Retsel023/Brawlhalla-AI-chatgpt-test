"""Inspect a local Brawlhalla installation without launching or changing it."""

import hashlib
from pathlib import Path
import re
import stat
import xml.etree.ElementTree as ET


DEFAULT_GAME_DIRECTORY = r"C:\Program Files (x86)\Steam\steamapps\common\Brawlhalla"
INSPECTED_SWF_SHA256 = "3b57541b92881ca8c0e6cb7b2f2d8661469e2c8734dbe8963dc72acd65bc30c5"
AIR_NAMESPACE = re.compile(r"http://ns\.adobe\.com/air/application/[0-9]+(?:\.[0-9]+)+\Z")
REQUIRED_FILES = {
    "executable": "Brawlhalla.exe",
    "swf": "BrawlhallaAir.swf",
    "manifest": "META-INF/AIR/application.xml",
}


def _file_check(path):
    check = {"path": str(path)}
    try:
        info = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return {**check, "status": "missing", "message": "Required file is missing."}
    except OSError as error:
        return {**check, "status": "unreadable", "message": str(error)}
    if not stat.S_ISREG(info.st_mode):
        return {**check, "status": "not_file", "message": "Expected a regular file."}
    return {**check, "status": "present", "message": "Required file is present."}


def _inspect_manifest(path, check):
    try:
        with path.open("rb") as stream:
            root = ET.parse(stream).getroot()
    except OSError as error:
        return {**check, "status": "unreadable", "message": str(error)}
    except (ET.ParseError, ValueError) as error:
        return {**check, "status": "invalid", "message": f"Cannot parse AIR XML: {error}"}
    namespace = ""
    if isinstance(root.tag, str) and root.tag.startswith("{"):
        namespace, _, name = root.tag[1:].partition("}")
    else:
        name = root.tag
    if name != "application" or not AIR_NAMESPACE.fullmatch(namespace):
        return {**check, "status": "invalid", "namespace": namespace,
                "message": "Expected an application element in the Adobe AIR namespace."}
    architecture = root.findtext(f"{{{namespace}}}architecture", default="").strip()
    content = root.findtext(
        f"{{{namespace}}}initialWindow/{{{namespace}}}content", default="").strip()
    diagnostics = []
    if architecture != "64":
        diagnostics.append(f"Expected architecture 64; found {architecture!r}.")
    if content != "BrawlhallaAir.swf":
        diagnostics.append(f"Expected entry content BrawlhallaAir.swf; found {content!r}.")
    return {**check, "status": "invalid" if diagnostics else "valid",
            "namespace": namespace, "architecture": architecture, "content": content,
            "message": " ".join(diagnostics) or "64-bit AIR manifest and SWF entry match."}


def inspect_install(directory, expected_swf_sha256=None):
    """Return JSON-safe file, manifest, and optional build-fingerprint checks.

    ``ready`` means the required files and AIR manifest are valid, and the SWF
    matches when an expected SHA-256 was supplied. It does not mean that the
    game runs, a live state provider exists, or the manifest's nominal version
    identifies the game build. Only the SWF and XML are opened, both read-only.
    """
    root = Path(directory).expanduser().resolve()
    expected = expected_swf_sha256
    expected_valid = expected is None or (
        isinstance(expected, str) and re.fullmatch(r"[0-9a-fA-F]{64}", expected) is not None)
    if isinstance(expected, str):
        expected = expected.lower()
    checks = {name: _file_check(root / relative)
              for name, relative in REQUIRED_FILES.items()}
    swf_match = None
    swf = checks["swf"]
    if swf["status"] == "present":
        digest = hashlib.sha256()
        try:
            with (root / REQUIRED_FILES["swf"]).open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as error:
            swf.update(status="unreadable", message=str(error))
        else:
            swf["sha256"] = digest.hexdigest()
            if expected is not None and expected_valid:
                swf["expected_sha256"] = expected
                swf_match = swf["sha256"] == expected
                swf.update(
                    status="match" if swf_match else "mismatch",
                    message=("SWF matches the expected build fingerprint." if swf_match else
                             "SWF differs from the expected build fingerprint; static findings are version-specific."))
            else:
                swf["message"] = "SWF fingerprint computed; no expected build fingerprint supplied."
    if not expected_valid:
        checks["expected_swf_sha256"] = {
            "status": "invalid", "message": "Expected SWF SHA-256 must be exactly 64 hexadecimal characters."}
    manifest = checks["manifest"]
    if manifest["status"] == "present":
        checks["manifest"] = _inspect_manifest(root / REQUIRED_FILES["manifest"], manifest)
    missing = [REQUIRED_FILES[name] for name, check in checks.items()
               if name in REQUIRED_FILES and check["status"] == "missing"]
    ready = all(check["status"] in {"present", "valid", "match"}
                for check in checks.values())
    return {"directory": str(root), "ready": ready,
            "status": "ready" if ready else "missing" if missing else "invalid",
            "missing": missing, "swf_match": swf_match, "checks": checks}
