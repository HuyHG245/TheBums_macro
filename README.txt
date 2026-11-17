THE BUMS - Attack on Titan: Revolution macro

OPTION A - MAKE ONE PROGRAM FILE (no console window)
  1. Install Python 3.10 or newer from python.org. Tick "Add Python to PATH".
  2. Double-click build.bat. The first build takes a few minutes.
  3. Start TheBums.exe. It opens only the program window, nothing else.
  (build.bat onedir makes a folder instead of one file. It starts faster and
   antivirus programs complain less. Run dist\TheBums\TheBums.exe.)

OPTION B - RUN FROM SOURCE
  1. Install Python 3.10 or newer (tick "Add Python to PATH").
  2. Double-click setup.bat once, then run.bat to start. No console window stays open.
  If something goes wrong, run  .venv\Scripts\python.exe main.py  in a terminal to see the messages.

BEFORE YOU PRESS START
  - Roblox must fill the screen: press F11 in Roblox (borderless fullscreen).
    The pictures are matched at screen size, so a small window is not recognised.
  - On Start the activity log says which monitor it watches and whether Roblox fills it.
  - To check what the macro sees: Settings > Configs > Asset override > Test, then open the game.
  - Default hotkeys: F8 start, F6 pause, F7 stop. You can change them in Settings.

GOOD TO KNOW
  - Opacity slider: footer of the main window, 30-100 %. It is remembered.
  - Data is saved in Documents\The Bums (settings, stats, configs, your own pictures).
    If the program stops because of an error, crash.log is written there too.
  - The only network use is the Discord webhook, and only if you enter a URL in Settings.
  - Windows Defender may warn about any program made with PyInstaller. Only allow a TheBums.exe
    you built yourself from these files. Never allow an .exe you downloaded from somewhere else.
  - Logo colours: copy the 3 files from logo_variants\<colour>\ into the image folder
    (overwrite), then build again. The default colour is Ember.
  - The image folder holds the pictures the macro looks for on screen. Do not rename or delete them.
