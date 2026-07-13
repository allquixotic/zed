---
title: Zed on Windows
description: "Get the latest stable builds via the download page. If you want to download our preview build, you can find it on its [releases p..."
---

# Zed on Windows

## Installing Zed

Get the latest stable builds via [the download page](https://zed.dev/download). If you want to download our preview build, you can find it on its [releases page](https://zed.dev/releases/preview). After the first manual installation, Zed will periodically check for install updates.

You can also build zed from source, see [these docs](https://zed.dev/docs/development/windows) for instructions.

### Package managers

Additionally, you can install Zed using winget:

```sh
winget install -e --id ZedIndustries.Zed
```

## Uninstall

- Installed via installer: Use `Settings` → `Apps` → `Installed apps`, search for Zed, and click Uninstall.
- Built from source: Remove the build output directory you created (e.g., your target/install folder).

Your settings and extensions live in your user profile. When uninstalling, you can choose to keep or remove them.

## Remote Development (SSH)

Zed supports remote development on Windows through both SSH and WSL. You can connect to remote servers via SSH or work with files inside WSL distributions directly from Zed.

For detailed instructions on setting up and using remote development features, including SSH configuration, WSL setup, and troubleshooting, see the [Remote Development documentation](./remote-development.md).

## Troubleshooting

### Zed fails to start or shows a blank window

- Check that your hardware and operating system version are compatible with Zed. See our [installation guide](./installation.md) for more information.
- Update your GPU drivers from your GPU vendor (Intel/AMD/NVIDIA/Qualcomm).
- Ensure hardware acceleration is enabled in Windows and not blocked by third‑party software.
- Check the Rendering Backend row in Settings to see whether Zed automatically fell back to CPU software rendering.
- Try launching Zed with no extensions or custom settings to isolate conflicts.

### Terminal issues

If activation scripts don’t run, update to the latest version and verify your shell profile files are not exiting early. For Git operations, confirm Git Bash or PowerShell is available and on PATH.

### SSH remoting problems

When prompted for credentials, use the graphical askpass dialog. If it doesn’t appear, check for credential manager conflicts and that GUI prompts aren’t blocked by your terminal.

### Graphics issues

#### Zed fails to open / degraded performance

Zed uses Direct3D 11 for hardware-accelerated rendering. If no compatible hardware GPU can be initialized, Zed automatically starts with its native CPU software renderer instead of a WARP adapter. Automatic fallback is retried on every launch and does not change your settings.

To use CPU rendering deliberately, select **Software (CPU)** under **Settings → Window & Layout → Window → Rendering Backend**, then restart Zed. You can also add this root-level user setting:

```json
{
  "rendering_backend": "software"
}
```

Set it back to `"auto"` to retry hardware rendering on the next launch. This setting is user-wide and cannot be set in project settings or settings profiles. Software mode uses opaque windows and reduces nonessential visual effects.

If Zed falls back to software rendering and you want to restore hardware rendering, check DirectX support by running:

```
dxdiag
```

This will open the DirectX Diagnostic Tool, which shows the DirectX version your GPU supports under `System` → `System Information` → `DirectX Version`.

In a virtual machine without GPU passthrough, Zed uses its CPU software renderer. The Rendering Backend settings row and Zed log report the active backend and any hardware initialization failure.
