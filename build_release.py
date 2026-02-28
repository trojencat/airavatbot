#!/usr/bin/env python3
"""
Build script for Airavat Windows Release
Creates a packaged release with daemon executable, webui, and configs
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
#!/usr/bin/env python3
"""
Build script for Airavat Release (cross-platform)
Creates a packaged release with daemon executable, webui, and configs

This script is used by the top-level `build_all.py` and can also be
invoked directly. It now detects the host platform and will copy
the generated executable regardless of platform-specific extension.
"""
import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path


def get_version():
    """Extract version from pyproject.toml"""
    pyproject = Path(__file__).parent / "pyproject.toml"
    with open(pyproject) as f:
        for line in f:
            if line.startswith('version = "'):
                return line.split('"')[1]
    return "unknown"


def detect_platform_tag() -> str:
    system = platform.system().lower()
    if system.startswith('win'):
        return 'windows'
    if system.startswith('darwin'):
        return 'macos'
    return 'linux'


def main():
    daemon_dir = Path(__file__).parent
    project_root = daemon_dir.parent
    release_dir = project_root / "release"

    version = get_version()
    platform_tag = detect_platform_tag()
    release_name = f"airavat-v{version}-{platform_tag}"
    release_path = release_dir / release_name

    print(f"\n⚙️  Building Airavat Release v{version} ({platform_tag})...")
    print(f"📁 Release directory: {release_path}\n")

    # Step 1: Clean previous builds
    if release_path.exists():
        print("🗑️  Cleaning previous build...")
        shutil.rmtree(release_path)
    release_dir.mkdir(parents=True, exist_ok=True)
    release_path.mkdir(parents=True, exist_ok=True)

    # Step 2: Build executable with PyInstaller
    print("🔨 Building executable with PyInstaller...")
    spec_file = daemon_dir / "airavat-daemon.spec"
    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(spec_file),
        "--onefile",
        "--distpath", str(release_path / "dist"),
        "--buildpath", str(release_path / "build"),
        "--specpath", str(release_path),
    ]

    result = subprocess.run(cmd, cwd=str(daemon_dir))
    if result.returncode != 0:
        print("❌ PyInstaller build failed!")
        return False

    # Step 3: Copy executable to release root (flexible for different OS builds)
    dist_dir = release_path / "dist"
    exe_candidate = None
    if dist_dir.exists():
        for p in dist_dir.iterdir():
            if p.name.startswith('airavat-daemon'):
                exe_candidate = p
                break

    if exe_candidate:
        target_name = 'airavat-daemon.exe' if platform_tag == 'windows' else 'airavat-daemon'
        print(f"📦 Packaging executable ({exe_candidate.name}) as {target_name}...")
        shutil.copy(exe_candidate, release_path / target_name)
        if platform_tag != 'windows':
            try:
                os.chmod(release_path / target_name, 0o755)
            except Exception:
                pass
    else:
        print(f"⚠️  Warning: No executable found in {dist_dir}")

    # Step 4: Copy config files
    print("📄 Copying configuration files...")
    configs = ["agent-config.json", "mcp-config.json", "ui-config.json"]
    for config in configs:
        src = daemon_dir / config
        if src.exists():
            shutil.copy(src, release_path / config)
        else:
            print(f"  ⚠️  {config} not found (will be created on first run)")

    # Step 5: Create README
    readme = release_path / "README.txt"
    readme.write_text(f"""Airavat Daemon v{version}

QUICK START:
1. Run the daemon executable to start the server
2. Open http://localhost:8920 in your browser
3. The daemon will run on port 8920

REQUIREMENTS:
- Python is not required (bundled in executable)

CONFIGURATION:
- agent-config.json    - LLM and agent settings
- mcp-config.json      - MCP server connections
- ui-config.json       - Theme and UI preferences

TROUBLESHOOTING:
If the server doesn't start:
- Make sure port 8920 is not in use
- Check if there's a firewall blocking it
- Run from command line to see error messages

For more info, visit: https://github.com/airavat/airavat
""")

    # Step 6: Cleanup build artifacts
    print("🧹 Cleaning up build artifacts...")
    if (release_path / "dist").exists():
        shutil.rmtree(release_path / "dist")
    if (release_path / "build").exists():
        shutil.rmtree(release_path / "build")

    print(f"\n✅ Release ready at: {release_path}")
    print(f"📦 Archive folder: {release_name}/")
    print(f"\n💾 To distribute:\n   Zip the entire '{release_name}' folder and share\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
