# The Bums Macro

**Windows source · Setup · Build · Update**

The Bums Macro is an independent keyboard/mouse automation application for Roblox: Attack on Titan Revolution. It is not affiliated with Roblox or the game. Use it in accordance with applicable game rules.

> **Status:** v1.3 source snapshot for Windows. Reconnect, the updater and several beta features still require Windows/Roblox end-to-end verification. **Background Mode is In Development and cannot be enabled.** The default Standard Mode remains available.

## Quick start — TheBumsMacro_v1.3 source ZIP

**This ZIP contains the full v1.3 source, not a prebuilt Windows EXE.** Extract to a separate folder, especially if you already have a working installation.

1. Install **Python 3.10+ on Windows** and check **Add Python to PATH**.
2. Double-click **`setup.bat`** once (internet connection required for dependencies).
3. Double-click **`run.bat`** to open the Beta application.
4. Optionally double-click **`build.bat`** to build `TheBumsMacro.exe` on Windows.

If the app does not open, run ` .venv\Scripts\python.exe main.py ` in a terminal from this folder to see any error messages.

**Beta caution:** Test with your existing setup backed up. Do not depend on the new reconnect, automatic update, or rollback during an unattended overnight session until tested on your Windows/Roblox installation.

## 1. Download the application

1. Open the [official Releases page](https://github.com/HuyHG245/TheBums_macro/releases/latest).
2. Expand **Assets** and download `TheBumsMacro.exe`.
3. Save the EXE in a folder you can write to, such as `Documents\TheBumsMacro`.
4. Launch it. Windows may request administrator consent because of the keyboard/mouse input requirements.

A prebuilt EXE does **not** require Python. Download only from the official repository.

## 2. Run from source

1. Install **Python 3.10+** for Windows and enable **Add Python to PATH**.
2. Download the repository with **Code → Download ZIP** and extract it, or clone it:

   ```powershell
   git clone https://github.com/HuyHG245/TheBums_macro.git
   cd TheBums_macro
   ```

3. Double-click **`setup.bat`** to create a Python environment and install dependencies.
4. Double-click **`run.bat`** to launch the app.

Please use **Windowed Screen, not Fullscreen** for Roblox. The optional Adaptive Resolution setting can capture the Roblox client area on supported Windows configurations.

## 3. Build the Windows EXE

Double-click **`build.bat`** to make the self-contained `TheBumsMacro.exe` in the project directory and `dist`.

For a folder-based distribution run:

```powershell
.\build.bat onedir
```

The build uses PyInstaller and embeds the `image/` templates, `guides/` and the local `updater.ps1` helper. Folder-based builds must remain together. The build may request UAC approval at launch; it does not bypass Windows security.

## 4. Updating the application

Click **Updates** in the main window to compare your installed version with the latest GitHub Release.

For a future release that ships the verified Update Manager, the app can download and verify a new EXE and install it without manual ZIP extraction. **It cannot install an update while a macro session is active, paused, reconnecting, or otherwise not explicitly stopped.** Downloads can be prepared in advance.

In **Settings → App → Update Manager**, choose whether to enable **Automatic Updates**, **Auto Download** and **Install automatically after Stop** or **Ask before installing**. Automatic Updates start **OFF** by default.

Update transactions retain the previous EXE for rollback if the newly installed EXE fails its startup check. Failed builds are recorded to avoid repeated automatic reinstalls. **Update History** shows the last 20 records. Real-world verification is required before depending on these capabilities.

If a release does not provide a verifiable Windows EXE asset with SHA-256, the application refuses automatic installation and offers the release page instead.

## 5. Personal settings and troubleshooting

Your personal data normally lives in **`Documents\The Bums`**, separately from the application executable. This includes configurations, statistics, user-created macros, diagnostics and local settings.

For common problems:

- Python missing: install Python 3.10+ and run `setup.bat` again.
- Build failed: check the terminal output and install dependencies with an internet connection.
- Screen recognition problems: use **Settings → Detection → Run Diagnostics** and save its local report.
- Unexpected macro errors: inspect the locally saved **Error Snapshots** (private PS code screens are not captured).
- Failed update: check **Update History** and see if the previous executable was restored.

Do not share private server codes, bot tokens, webhooks, or your personal configuration files publicly.

## 6. Official links

- [Repository](https://github.com/HuyHG245/TheBums_macro)
- [Latest release](https://github.com/HuyHG245/TheBums_macro/releases/latest)
- [Issue tracker](https://github.com/HuyHG245/TheBums_macro/issues)

**The Bums Macro — keyboard/mouse automation only.**

## Hybrid AI (Beta) — source development option

A local, CPU-only Hybrid AI verification integration is available in this
working source tree. **Standard/OpenCV recognition is still the default**;
existing `image/` templates and all gameplay macros are preserved. The Beta
option only considers borderline candidates in Retry, Loot and Image Reconnect.
It uses no API or cloud inference. Existing OpenCV recognition images remain unchanged.

**Background Mode — In Development:** The toggle in **Settings → Roblox** is
visible but disabled (cannot be switched on). It is disabled for gameplay and
old saved settings/presets cannot re-enable it. Windows testing showed that
Roblox could be captured while unfocused but did not accept tested keyboard,
click or camera messages. **Standard Mode** is the supported choice. The
experimental Background Input Diagnostics interface and module have been
removed from this source release. Ordinary **Settings → Detection → User
Diagnostics** remains available.

**No trained, gameplay-validated ONNX model is bundled.** The Hybrid AI option
requires a validated model and optional CPU ONNX Runtime; otherwise Standard
OpenCV recognition is used. See [models/README.md](models/README.md) for the
expected model and manifest. Use `setup-hybrid-ai.bat` optionally; `setup.bat`,
`run.bat` and `build.bat` retain Standard behavior.

## Settings navigation and session tools (development build)

**Settings** replaces the Dashboard inside the same main window and provides a
left sidebar: **Keybinds, Macros, Recovery, Roblox, Detection, Discord, App**.
Select **Settings → Macros → Session Analytics** for an in-page overview of
recorded sessions, completed and failed runs, completion rate, reconnects,
errors, a recent-session chart and session list. The local journal is bounded;
these are the *recorded events* available on this machine, not guaranteed
all-time server statistics. A last session with 0 runs can be normal even when
the totals from earlier sessions are nonzero.

Select **Settings → Macros → Config Presets** to save, apply, duplicate, rename
and delete named presets without opening another window. New local presets
capture the full `settings` snapshot, including all macro timing overrides,
Recovery, Detection, Discord, Roblox, Keybinds and App options. **Local preset
storage also contains private PS codes and credentials**: treat the saved
`config_presets.json` as sensitive, just like your normal settings file.
Older macro-only presets remain readable. Stop the macro before applying or
changing presets; a second click confirms overwrite and delete.

**Offline preset sharing:** select a saved preset and click **Generate Share Code**
then **Copy Code**. Send the self-contained `TBP1...` text to the recipient;
no account, server or API is required. They paste it into the same page and
click **Import Code**, then can select **Apply**. Repeated imports receive
independent names. The code includes portable options but *never* includes
private PS codes, Discord bot tokens, webhook URLs, Discord user/channel IDs,
window geometry or a device's retry-scale cache. Import never auto-applies.
Applying an imported preset disables Auto Reconnect until checked locally;
existing private credentials on the recipient's computer are not overwritten.
Code checksums detect accidental corruption, **not impersonation**; accept codes
only from people you trust. Codes may be long because they contain the settings.
Custom macro script files and replacement recognition images are not settings
and are not included in the share code.

**Image Reconnect:** the original Reconnect-button flow remains supported.
For a Leave-only disconnect, OpenCV confirms the Leave dialog and button,
then accepts either Roblox Home **or directly the Attack on Titan Revolution
game detail page**; the correct route proceeds through the blue triangle Play
button to Slot A and the existing server/mission steps. These branches have
been simulated without Windows; actual Roblox behavior must still be tested.

**Settings → Detection** contains recognition settings and diagnostics.
**Settings → Roblox** contains monitor selection, adaptive resolution,
Performance Mode and **Background Mode (In Development, disabled)**. Keep Roblox
in **Windowed** mode for normal macro use.

**Testing status:** The inline Settings pages have passed a virtual-display
Tkinter smoke test and pure-data tests in a non-Windows development environment.
Actual Roblox input, capture, updates, notifications and crash recovery still
need Windows-level verification before being considered production ready.

## Stability / Shared Scan (v1.3)

**Background input remains unavailable.** The Background Mode switch is labeled
*In Development*, is non-interactive, and cannot be enabled using old
configuration files or imported presets. User-triggered Background Input
Diagnostics and the unused experimental diagnostic module are not distributed.

**Shared Screenshot / Smart Scan:** The passive Retry and Reconnect watchdog
continues to use a single capture per polling cycle. Its normalized BGR frame
can now also be reused by the foreground *non-decisive* Retry and Disconnect
scans if it is less than 180 ms old and matches the current capture geometry.
Any Retry verification just before clicking, mission start verification and
Reconnect step/state verification still take a **fresh capture**, never the
shared cache. Performance Mode adjusts polling frequency; pausing and active
Reconnect also reduce the passive scan rate. Diagnostic reports show the count
of shared frames published and reused. This is a conservative optimization,
not a complete replacement of every screenshot call in the macro.

**Macro Health Monitor:** Script-stage/step changes are tracked separately
from countdown redraws. If a running mission shows no step change for 210
seconds, or a Reconnect stage shows no step change for 130 seconds, the monitor
logs one warning, records a health event and attempts an existing protected
error snapshot. Idle waiting and Pause are excluded. This diagnostic **does not
invent recovery clicks or restart Roblox**, and is not proof that the game
responded to any key. Existing mission failsafe, Retry, and Reconnect logic
remains responsible for actual recovery.

**Validation:** Existing recognition tests and additional in-memory
state/cache/safety tests were run on Linux. Windows/Roblox gameplay
compatibility has not been established for the new performance changes.


## Settings & Analytics UI responsiveness (2026-11-22)

Settings is now a page inside the main application window. Returning to the
Dashboard hides the existing Settings widgets instead of destroying and
recreating all nine pages. Switching tabs preserves scroll position and avoids
full-page slide redraws. Rebuilding the Settings interface is still intentional
when the theme changes. Session Analytics avoids redrawing unchanged history,
and hovering a bar in **Results by Session** displays that session's Completed,
Failed, total results, macro and start time directly on the chart (no new
window). These behaviors were tested on a Linux virtual Tk display; final
rendering on Windows should be checked during normal beta use.

### Private Server Code (PS Code)

The PS Code field in Recovery > Reconnect accepts mixed-case letters and digits.
Use **Show PS Code** to visually check the exact code; it starts masked.
PS Code is saved locally and excluded from shared preset codes and diagnostics.
Image Reconnect now types each alphanumeric character with explicit key-down/up
and a short delay to reduce missed characters on Roblox. Entry still requires
Roblox to have focus in Standard Mode.
