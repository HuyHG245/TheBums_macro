# The Bums Macro

**Release:** `TheBumsMacro_v1.1`  
**Game:** Roblox — Attack on Titan: Revolution  
**Repository:** https://github.com/HuyHG245/TheBums_macro

The Bums Macro is a community-made **keyboard/mouse automation** application. It captures the Roblox screen, recognizes UI images and sends configured inputs. It does not require game-memory reading or code injection. It is not an official Roblox or Attack on Titan: Revolution product; using automation may be subject to the game's rules.

## Features

### Built-in macros

| Mode | Purpose |
| --- | --- |
| Shiki | Mission automation with Shiki-specific ability handling. |
| Universal | General mission routine with directional actions, camera movement and Shift handling. |
| Blood Awakening | Mission routine with Blood Awakening and improved Colossal Shift confirmation. |
| Female Raid | Raid automation with cutscene handling, chest positions and Retry support. |
| Attack Titan Raid Fritz (Not Recommended) | Fritz-based raid opening, directional dashes, automatic cutscene skip and Colossal combat. |
| Attack Titan Raid Blood Awakening | Acoustic Shells → forward Dash → Colossal Shift; then shared Colossal combat and chest handling. |

The two Attack Titan modes require **Colossal Titan** and use on-screen skill templates, including `acoustic_shells.png` and `heatwave.png`. Colossal Heatwave is controlled using its two-stage visibility/cooldown cycle; other Colossal skills operate on separate schedules. Raid chest options support the left Raider's Chest and optional right Emperor's Trove.

### Custom macros

- Create, record, edit, validate and run user-defined `.tbmacro` scripts.
- Commands include `press ability1`–`ability5`, `wait`, `hold`, `spam`, `dash`, `backdash`, `strafe`, mouse movement/clicks, image checks, loops, conditions and Retry/loot actions.
- The **Guide** button in the Custom Macro Editor opens a command reference with timing examples.
- Built-in mode Guides and skill-layout images are included; guide content is stored in `guides/*.txt` and formatted by the app.

### Automation, detection and recovery

- On-screen detection for mission/raid readiness, results, Retry, rewards, cutscenes and selected ability icons.
- Persistent Retry monitoring designed to avoid a stalled result screen and avoid double counting.
- Raid chest Pick Spot configuration, Auto Skip and image-detection overrides.
- Loot reporting for Scrolls, Vessels, Fragments and Serums, including Attack Titan-specific **Primordial Devil** and **Attack Serum**.
- Saved session/statistics counters and detailed activity logs.
- Reconnect and failsafe options for supported flows.

### User interface and connectivity

- App branding: **The Bums Macro**, with a note that it is keyboard/mouse automation only.
- Configurable hotkeys/keybinds, per-macro Edit Configs with persistence and Reset.
- Opacity slider (**30–100%**), Always on Top, themes and compact stats-bar mode.
- Roblox monitor selection/automatic monitor detection.
- **Adaptive Resolution (Beta)** is **OFF by default**. When enabled on Windows, it prefers capturing Roblox's client area and remaps template-match coordinates; the legacy full-monitor detector is used when disabled.
- GitHub update checks (Releases and repository commits), Discord webhook/bot options and result reports.

## Requirements

- Windows 10/11 and Roblox.
- Python **3.10+** for running or building from source.
- Dependencies listed in `requirements.txt`.

## Run from source

1. Install Python, enabling **Add Python to PATH**.
2. Run `setup.bat` once to create the virtual environment and install packages.
3. Run `run.bat`, or run `.venv\Scripts\python.exe main.py` from a terminal for troubleshooting.
4. In **Settings → Configs**, choose your macro, read its Guide and configure chest positions or timings where applicable.
5. Keep Roblox in focus when testing inputs. Default hotkeys: **F8 Start**, **F6 Pause**, **F7 Stop**.

User-specific settings and history are stored under **Documents\The Bums**, outside this source repository in the normal Windows configuration. When storage there is unavailable, the app may fall back to its own folder: keep the `.gitignore` in place and review changes before publishing.

## Build an EXE

Run `build.bat` to build a single `TheBums.exe`. `build.bat onedir` makes a directory-based build instead. The build uses PyInstaller with `--uac-admin`; Windows shows a normal UAC consent prompt at launch. Administrator rights can help with input to an elevated Roblox window, but do not guarantee compatibility or immunity to detection errors.

Build outputs, installed Python environments, local configuration, logs, screenshots, user macros and credentials are excluded from Git by `.gitignore`.

## Adaptive Resolution (Beta)

The stable/default capture mode remains **Legacy (OFF)**. Enable Adaptive Resolution only when you want to test capture from the Roblox client region. The app's **?** tooltip explains the mode in Settings.

Known limitations: different Roblox HUD scales, DPI settings, nonstandard window layouts, multi-monitor setups and user-specific chest/reconnect coordinates may still require calibration. For stable macro operation, test Legacy mode first and use **Settings → Fix Image Detection** to verify templates when needed.

## Security and privacy

- Do **not** commit your Discord webhook URL, bot token, personal JSON configuration, screenshots, custom macros or debug captures.
- The `.gitignore` prevents common accidental *new* files from being staged, but it cannot remove secrets from files already tracked or from Git history. Inspect `git status` / `git diff --cached` before every push.
- If a credential was ever committed, revoke/rotate it and clean Git history before publishing; a later `.gitignore` entry alone is not enough.
- Only run EXE files built from source you trust. Be careful with administrator prompts.

## Changelog — TheBumsMacro_v1.1

The following is a comparison with the **original The Bums ZIP supplied for this project**, not a claim that every item is new compared with every historical GitHub release.

### New features since that original ZIP

- Added **Attack Titan Raid Fritz (Not Recommended)** and **Attack Titan Raid Blood Awakening**, including distinct opening sequences and a shared Colossal combat state machine.
- Added Shift confirmation using `heatwave.png`, Heatwave's two-cycle behavior, phase cutscene handling, raid chest processing and Attack Titan-specific loot templates.
- Added **Custom Macro Editor**, recorder, script validation, configurable command language and a formatted in-app command reference.
- Added **Recovery / Reconnect** options and stronger Retry monitoring/retry-screen handling.
- Added **Adaptive Resolution (Beta)**, disabled by default, with a nearby explanatory **?** tooltip.
- Added GitHub **Updates** checks and optional Discord bot workflow alongside the existing webhook functionality.
- Added per-macro **Edit Configs** behavior, with saved values and filtered shared controls.
- Added independent Scroll display and Attack Titan-specific Primordial Devil / Attack Serum tracking.

### Improvements and fixes

- Retained and refined the original **opacity slider**, automatic Roblox monitor detection, themes, stats and Discord reporting.
- Improved Blood Awakening Shift reliability and more conservative loot matching.
- Added diagnostic logging for input steps, image matching and Retry recovery.
- Added scalable image matching for some critical raid ability templates and explicit directional double-tap timing.
- Redesigned Custom Macro Guide rendering and made attack-raid Guides identify their required builds.
- Updated in-app branding to **The Bums Macro** and added the discreet automation disclaimer.
- Added Windows EXE **UAC administrator prompt** support in `build.bat`.
- Cleaned the GitHub source distribution: removed obsolete experiment notes, temporary build files and stale version readmes; kept the working Guide content in `.txt` files so **README.md is the only Markdown file**.

### What was already in the original ZIP

The original The Bums release already contained **Shiki, Universal, Blood Awakening and Female Raid**, built-in guides, a loot/stats UI, Discord webhook notifications, multiple themes, Always on Top, **opacity control**, Roblox monitor detection and crash logging. These are **retained capabilities**, not claimed as newly invented in this release.

## Source and licensing

This project incorporates code from more than one source baseline. Before publishing or redistributing it, ensure you have permission for each third-party contribution and preserve any copyright/license notices that the applicable licenses require. The repo's own branding does not override those obligations.
