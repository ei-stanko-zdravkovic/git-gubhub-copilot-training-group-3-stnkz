#!/usr/bin/env python3
"""
Cross-platform Synth CLI installer.

Downloads and installs Synth v0.6.9 for macOS or Windows.
"""
import os
import platform
import subprocess
import sys
import tempfile
import urllib.request
import tarfile
import shutil
from pathlib import Path

SYNTH_VERSION = "0.6.9"
GITHUB_RELEASE_BASE = f"https://github.com/shuttle-hq/synth/releases/download/v{SYNTH_VERSION}"

# Platform-specific download URLs
DOWNLOAD_URLS = {
    "Darwin": f"{GITHUB_RELEASE_BASE}/synth-macos-latest-x86_64.tar.gz",
    "Windows": f"{GITHUB_RELEASE_BASE}/synth-windows-latest-x86_64.exe",
}


def get_install_dir() -> Path:
    """Return the installation directory (~/.synth/bin)."""
    home = Path.home()
    return home / ".synth" / "bin"


def download_file(url: str, dest: Path) -> None:
    """Download a file from url to dest."""
    print(f"⬇️  Downloading from {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"✓ Downloaded to {dest}")


def install_macos() -> bool:
    """Install Synth on macOS."""
    install_dir = get_install_dir()
    install_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tarball = Path(tmpdir) / "synth.tar.gz"
        download_file(DOWNLOAD_URLS["Darwin"], tarball)

        print("📦 Extracting archive...")
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(path=install_dir)

    synth_bin = install_dir / "synth"
    synth_bin.chmod(0o755)
    print(f"✓ Synth installed to {synth_bin}")

    # Add to PATH in .zshrc (default macOS shell)
    zshrc = Path.home() / ".zshrc"
    export_line = f'export PATH="$HOME/.synth/bin:$PATH"'
    if zshrc.exists():
        content = zshrc.read_text()
        if ".synth/bin" not in content:
            with zshrc.open("a") as f:
                f.write(f"\n# Synth CLI\n{export_line}\n")
            print(f"✓ Added Synth to PATH in {zshrc}")
        else:
            print(f"ℹ️  PATH entry already present in {zshrc}")
    else:
        with zshrc.open("w") as f:
            f.write(f"# Synth CLI\n{export_line}\n")
        print(f"✓ Created {zshrc} with Synth PATH")

    return True


def install_windows() -> bool:
    """Install Synth on Windows."""
    install_dir = get_install_dir()
    install_dir.mkdir(parents=True, exist_ok=True)

    synth_exe = install_dir / "synth.exe"
    download_file(DOWNLOAD_URLS["Windows"], synth_exe)
    print(f"✓ Synth installed to {synth_exe}")

    # Add to User PATH via setx
    install_dir_str = str(install_dir)
    current_path = os.environ.get("PATH", "")
    if install_dir_str.lower() not in current_path.lower():
        print("🔧 Adding Synth to User PATH...")
        # Use PowerShell to append to User PATH
        ps_cmd = (
            f'[Environment]::SetEnvironmentVariable("Path", '
            f'[Environment]::GetEnvironmentVariable("Path", "User") + ";{install_dir_str}", "User")'
        )
        subprocess.run(["powershell", "-Command", ps_cmd], check=True)
        print("✓ Added Synth to User PATH (restart terminal to take effect)")
    else:
        print("ℹ️  Synth directory already in PATH")

    return True


def verify_installation() -> bool:
    """Verify Synth is installed and runnable."""
    synth_bin = get_install_dir() / ("synth.exe" if platform.system() == "Windows" else "synth")
    if not synth_bin.exists():
        print(f"✗ Synth binary not found at {synth_bin}")
        return False

    try:
        result = subprocess.run([str(synth_bin), "version"], capture_output=True, text=True)
        version_output = result.stdout.strip() or result.stderr.strip()
        print(f"✓ Synth version: {version_output}")
        return True
    except Exception as e:
        print(f"✗ Failed to run synth: {e}")
        return False


def main() -> int:
    system = platform.system()
    print(f"🖥️  Detected OS: {system}")

    if system == "Darwin":
        success = install_macos()
    elif system == "Windows":
        success = install_windows()
    else:
        print(f"✗ Unsupported OS: {system}. Only macOS and Windows are supported.")
        return 1

    if not success:
        return 1

    print("\n🔍 Verifying installation...")
    if verify_installation():
        print("\n✅ Synth installation complete!")
        if system == "Darwin":
            print("   Run `source ~/.zshrc` or open a new terminal to use `synth`.")
        else:
            print("   Open a new terminal window to use `synth`.")
        return 0
    else:
        print("\n⚠️  Installation succeeded but verification failed.")
        print("   Try opening a new terminal and running `synth version`.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
