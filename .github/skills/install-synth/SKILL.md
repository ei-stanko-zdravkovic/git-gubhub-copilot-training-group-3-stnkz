---
name: install-synth
description: Instructions for installing the Synth CLI (synthetic data generator) on macOS or Windows.
---

# Synth CLI Installation

When the user wants to install Synth, **do not** run raw curl/wget commands. Use the included Python script which handles cross-platform logic (Windows/macOS) automatically.

## Standard Install Procedure

Run the following commands in order:

1. **Navigate to the Repository Root**:
   ```bash
   cd /path/to/airlst-github-copilot-training-group-3
   ```

2. **Run the Install Script**:
   This script detects the OS, downloads the correct binary, and adds Synth to PATH.

   ```bash
   python3 .github/skills/install-synth/install_synth.py
   # OR on Windows (if python3 is not recognized):
   python .github/skills/install-synth/install_synth.py
   ```

3. **Reload the Shell** (macOS/Linux only):
   ```bash
   source ~/.zshrc   # or source ~/.bashrc
   ```

4. **Verify Installation**:
   ```bash
   synth version
   ```
   Expected output: `synth 0.6.9`

## Platform Notes

### macOS
- Installs to `~/.synth/bin/synth`
- Adds `~/.synth/bin` to PATH in `~/.zshrc`
- Uses x86_64 binary (works on Apple Silicon via Rosetta)

### Windows
- Installs to `%USERPROFILE%\.synth\bin\synth.exe`
- Adds the directory to User PATH environment variable
- Requires restarting the terminal after installation

## Troubleshooting

### synth: command not found
- Ensure the shell was restarted or sourced after installation.
- On Windows, open a **new** terminal window after install.

### Permission Denied (macOS)
- Run `chmod +x ~/.synth/bin/synth` manually if needed.

### Download Fails
- Check internet connection.
- Verify the release URL is reachable: https://github.com/shuttle-hq/synth/releases/tag/v0.6.9
