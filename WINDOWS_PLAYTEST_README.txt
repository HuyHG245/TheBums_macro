THE BUMS MACRO - GRID v11.4 - WINDOWS PLAYTEST

This is a test build of the existing source, not a stable release.
Extract to a new folder; do not overwrite the previous stable installation.
Run run-grid-v11.4.bat for the Grid reward pipeline (sets THEBUMS_GRID114=1).
The launcher checks dependencies and may install CPU PyTorch; Internet may be required.
Use your existing settings, enable Auto Reconnect and Auto Skip as appropriate.

Changes under test:
- Attack and Female Blood Awakening after-shift scheduler: 4 > 5 > 3 > 1.
- Heatwave double cooldown, activation confirmation and retry; HUD interruption safety.
- Seismic Shock visual readiness and retry; real cooldown/activation still needs gameplay validation.
- Reconnect detection fallback for idle Error 278 dialog.
- Reward temporal confirmation, deduplication and unknown values (?).

Offline verification in packaging session: 76 source tests passed; 13 reference temporal tests passed.
All 131 original baseline files present; 7 packaged baseline model assets unchanged.
The experimental fine-tuned candidate is not deployed or included.
No Windows gameplay test or new ONNX compatibility verification was performed.

Playtest checklist:
1. Attack Blood and Female Blood: observe priorities after shift; Heatwave first/second cooldown and retry.
2. Record Seismic Shock before activation, immediately after activation, while cooling down, and ready again.
3. Female cutscene / Auto Skip, free chest before Finish, retry if enabled.
4. If idle Error 278 naturally occurs, verify Reconnect (do not test Error 279).
5. Compare Webhook Gold/Gem/Perk with OBTAINED and check session totals.
Send recording and Activity logs for failures.

HOTFIX: BLOOD HUD / HEATWAVE / SHIFT (2026-11-26)
This package supersedes the supplied Windows Playtest ZIP for the next live test.
- Form is confirmed from two separate frames of the Colossal slot-2/3 HUD,
  independently of whether Heatwave is ready. The confirmed frame is handed
  directly into combat. No additional Skill 5 Shift press during confirmation.
- Heatwave is observed in its own slot, with icon shape and gold brightness;
  READY / COOLDOWN / UNKNOWN are distinct. UNKNOWN never counts as cooldown.
- Heatwave still requires two cooldown cycles before reactivation; resets only
  on confirmed activation. Priority 4 > 5 > 3 > 1 retained when eligible.
- One immediate attempt and one rapid retry; subsequent failed retries paced
  at >=1.5s so other skills can run. The same anti-starvation rule applies to
  Seismic Shock. No combat inputs in detected cutscenes or chest panels.
- Native and editable Blood Awakening openings use the same new Shift path.
  Fritz legacy combat, Female chest logic, reconnect and Grid reward pipeline
  are preserved. No ONNX retraining or model changes in this hotfix.

Offline verification: 102 tests passed, Python compileall passed.
Replay of provided 49.355s Windows clip at 5 samples/s: 247 samples.
  0.00-19.79s: no Colossal form (99 samples)
  19.80s: first sampled confirmed-form HUD (ready Heatwave)
  37.99s: first sampled dim Heatwave/cooldown
  47.99s: first sampled ready Heatwave again
  98 ready observations, 50 cooldown observations, 0 unknown after form.
  Median HUD observer runtime ~2.37 ms on this Linux environment (not the
  whole macro loop and not a Windows performance benchmark).
Real cooldown/activation of Seismic Shock and the complete second Heatwave
cycle are not visible in the clip; tested synthetically only. The detector
was calibrated on one recording; other UI themes/graphics modes and live
Windows input latency still need testing. No claim of a successful live raid.

NEXT WINDOWS TEST: use the same UI scale and existing configuration first.
Record from just before Shift until both Heatwave cooldown cycles finish.
Check first Skill 4 as soon as Colossal form appears, then skill 5/3/1,
confirm no Skill 4 after first recovery, reactivate only after second.
Also check cutscene and chest/Finish/Retry behavior. Save Activity logs.

RAID IMAGE RECONNECT HOTFIX (2026-11-26)
- Settings > Recovery > Reconnect > Configure: choose Image Recognition (Beta),
  select the server mode, enter PS Code if needed, then enable Reconnect.
- Attack Titan Raid (Fritz/Blood): reuse existing Mission reconnect through
  Lobby/Play -> RAIDS -> verify default ATTACK TITAN -> CREATE -> START.
- Female Raid: same route, then FEMALE TITAN -> verify selected gold border
  and no Attack heading -> CREATE -> START.
- Uses the same Acoustic Shells Skill 2 HUD gate as the normal raid routine.
- Three original lobby/raid captures supplied the new templates. Actual Female
  selected screen and raid room START were not supplied: selected-state and
  START verification are tested offline with simulated screens; Windows live
  test is still required. The existing room START template is reused.
- Existing Mission navigation, Coordinate Recognition, skill scheduler and
  model weights are unchanged. No need to set coordinates in image mode.

RAID OPENING / CUTSCENE RUNTIME HOTFIX (2026-11-26)
- Windows user confirmed Female Raid image reconnect reached the correct raid.
- Fixed missing local configuration accessor in native Blood Awakening
  opening. Both Attack Blood and Female now construct ColossalHUD with the
  active family's heatwave_threshold before entering Skill 5 Shift.
- After verified Raid START during image reconnect, enable the existing
  E Skip handler in the HUD wait only. No skip inputs in lobby navigation.
- Both ordinary and reconnect raid start gates require the Skip prompt to
  be absent (including 350 ms after last sighting) and Acoustic Shells visible.
- Expanded lower-right skip search with coarse/refined multi-scale matching.
  Auto skip setting and configured repeat interval are still honored.
- Regression: 132 tests passed, including native opening calls for both
  Blood families, simulated skip prompts, reconnect HUD gating, and negative
  crop from the latest Female Raid combat screenshot. Actual intro cutscene
  was not included in that screenshot; Windows validation remains required.
- No edits to the reconnect navigation module, scheduler, or model assets.

ERROR 256 REJOIN HOTFIX (2026-11-26)
- Image Recognition checks the verified English Error 256 dialog body, title
  and dark Leave label before routing to the existing Leave navigator.
- Error 278 continues to use Reconnect; Leave-only dialogs retain their route.
- Error 256: Leave -> AOT:R Home card (if needed) -> Play -> Slot A, then
  resume existing server/destination selection and HUD confirmation.
- Tested with real Error 256 dialog crop, resized/cropped captures, negative
  cases and simulated Home/detail transitions. 139 offline tests passed.
- Live Windows Error 256 recovery remains to be verified. Other unknown
  two-button errors are not automatically assigned to Leave by this change.

GRID v11.5 QUALITY GUARD (2026-11-26)
The v11.4 runtime/model baseline is retained. Only Grid currency handling is
changed to skip already-ineligible crops before inference. Existing launcher
run-grid-v11.4.bat is intentionally unchanged. See grid115_handover for test
results, comparison logs and integration limits. 150 macro tests passed; no
Windows gameplay test performed for this package. Extract separately.

FAILSAFE HOTFIX (2026-11-26, on Grid v11.5 Quality Guard)
- Female Blood Awakening and both Attack Titan native/edited Raid routines
  now check the fixed Raid deadline from the foreground tick. Female's existing
  180s total / 180s after legacy sequence condition is unchanged; Attack uses
  the same 180s fixed Raid default (new policy; not a user-specified timeout).
  All Mission and Custom Macro modes retain their Settings > Recovery timeout.
- The same Mission rescue procedure is used: release held keys, spam Skill 2
  for 5s; the 15s wait begins 2s after spam starts (17s total before optional
  Look Down), then click the configured Failsafe spot every 0.5s, checking Retry.
- Existing Retry/Disconnect watchdogs take precedence. Check the correct Raid
  chest/Finish handler before Eject and during rescue. No Eject on a verified
  chest screen; stop combat on two matching result-title observations.
- A title is only a passive stop/wait signal, not authority to click. Retry
  still requires original matching thresholds and fresh frame confirmation.
  If Retry cannot be verified, remain hands-off; no new guessed Retry clicks.
- Terminal check is rate-limited to 0.5s with coarse ROI screening followed
  by full-size confirmation. Health Monitor stays a passive 210s warning.
- Offline: 194/194 source tests (150 previous + 44 new); Python compileall PASS.
  On 20 available historical terminal screenshots (15 completed, 5 failed),
  heading diagnostic matched the expected outcome including the latest dim
  RAID FAILED screenshot. Not an independent accuracy or Windows benchmark.
- All source model assets and OpenCV templates are byte-identical to v11.5.
  No AI retraining, no ONNX change, no changes to GitHub main.

WINDOWS PLAYTEST: extract into a NEW folder, use the original launcher
run-grid-v11.4.bat (filename intentionally retained). First check normal
Female/Attack wins and chest handling; then observe a naturally slow/stalled
raid and confirm Failsafe logs after 180s (not 210s). If the game already shows
FAILED/COMPLETED, expect verified Retry or hands-off waiting, not Eject.
Check F7/Pause and Auto Reconnect. Never force disconnections for this test.
Do not overwrite the previous installation until Windows validation succeeds.

CHEST RECOVERY HOTFIX (2026-11-26, on v11.5 Failsafe Hotfix)
- Female image-based left OPEN: at most five actual click attempts, spaced
  2.5s apart, with a 30s stage deadline even if the panel is not found.
  Right OPEN remains opt-in and receives at most one click to avoid wasting
  Emperor Keys. A failed confirmation is logged as unconfirmed, not opened.
- Move, settle 150ms, then match the same control on a fresh frame before
  clicking. A missing or shifted control is not clicked. OPEN disappearance
  must persist on a visible chest panel; missing panel is not proof of opening.
- Native Female and Attack FINISH: at most five verified attempts, spaced
  2.5s apart; up to 30s to locate/click FINISH, then one separate 30s Retry
  window from the first successful input delivery. These replace the native
  Finish click-spam duration/interval policy; the custom command retains
  its configured bounded FINISH burst. The original Retry verification remains.
- Attack Fritz/Blood and legacy/custom coordinate-based chest routes retain
  their configured chest positions and finish_wait; no burst is sent without
  a visible panel. Paid chest changed to at most one click for key protection.
  Panel presence does not independently verify a user-configured coordinate.
- No combat Eject while in chest handling. Stop/pause and real disconnect
  detection remain active. On timeout: final verified Retry check, log and
  diagnostic snapshot; use existing Auto Reconnect only if enabled and ready,
  otherwise stop safely. This is recovery, not a guarantee of opening the chest.
- The 180s Raid combat failsafe and Mission timeout settings are unchanged.
- Offline regression: 218/218 tests (194 prior + 24 new), compileall PASS.
  New tests simulate persistent OPEN, missing/moving panels, paid chest,
  Retry/disconnect/stop, late FINISH and recovery with reconnect disabled.
- No live Roblox Windows test. The initial click delivery failure in the video
  is still unverified; possible input/focus/coordinate or matching issues.
  The video does not show the chest screen at the time of the failure.
- Extract to a new folder; launch using run-grid-v11.4.bat (unchanged).
  First test an ordinary Female victory with free chest, then optional paid
  chest if desired; confirm normal Attack chest handling and Auto Retry.
  If an OPEN is stuck, inspect Chest Recovery log and error snapshot. Do not
  overwrite the previous installation until Windows tests succeed.

GRID v11.5 CHEST INPUT & REJOIN HOTFIX
------------------------------------
Observed: CHESTS remained visible while Image Reconnect repeatedly searched for
Slot A. This was a synthetic chest timeout, not a verified Disconnect dialog.
- Normal Windows chest input activates the verified Roblox process, moves and
  nudges the cursor, reacquires the button, then checks focus, cursor and the
  HWND under the click. No input is sent to a covered/incorrect target.
- A chest timeout uses a real game relaunch for Image Reconnect if the current
  screen is not already a verified Disconnect/Home/game detail/Slot A state.
  If that rejoin fails, stop instead of entering endless Waiting for raid.
  A relaunch can forfeit unopened rewards; it is only a bounded last resort
  when Auto Reconnect is enabled. Genuine disconnect route is unchanged.
- Start on an existing CHESTS screen: require FINISH plus CHESTS heading in
  two frames; finish the previous raid without spending another key. Manually
  open any desired remaining chests before Start if needed.
- Use Settings > App > Compact mode and turn off Always on top if the macro
  window covers the left chest; keep Roblox in windowed mode. The Win32 focus
  and cursor behavior still requires live validation.
- Offline pytest: 232 passed (218 previous tests plus 14 new). Existing
  timeout/retry/paid-chest limits, combat and AI assets are unchanged.

=== v1.4 BETA / HEATWAVE CUTSCENE UPDATE ===
Base: Grid v11.5 Chest Input & Rejoin Hotfix. This is source, not an EXE.
Run run-grid-v11.4.bat to test the existing Grid v11.5 analytics path.
The filename retains its original name for compatibility.
- Female Raid and both Attack Titan Raid modes now share the double-cooldown scheduler.
- Fritz keeps its configured skill intervals and timer-based Skill 5. Lower-skill
  priority now follows the shared scheduler (5, 3, 1), one input per tick.
- If HUD/cutscene interrupts an unconfirmed Heatwave input, do not toggle it
  again based only on a ready icon. Require fresh dark/bright/dark/bright evidence.
- If a complete cooldown cycle was hidden, Heatwave may abstain until the next
  raid rather than guess. Skills 5/3/1 continue on valid combat HUD.
- If the second cooldown was already verified, a fresh stable ready icon
  permits Heatwave to resume. Combat timeout, chest handling and reconnect unchanged.
- Offline: baseline 232/232; revised suite 238/238, including six added tests and
  three updated tests for intentional safety behavior. No live Roblox test or EXE build.

Windows checks (especially Female Raid):
1. Start a new raid and observe initial Shift and Skill 4 activation.
2. Watch the first ready icon: Skill 4 must NOT be sent here.
3. Observe second cooldown and next ready icon: Skill 4 may be sent after verification.
4. Observe a mid-raid cutscene and confirm no blind Skill 4 presses during/after it.
5. Confirm skills 5/3/1 resume when the combat HUD returns; note if Heatwave stays held.
6. Check chests, Finish, Auto Retry, failure and the existing 180s raid failsafe.
7. Repeat with Attack Fritz and Attack Blood. Send activity log/video if any issue.

V1.4 HYBRID AI OPT-IN
Settings > Detection > Recognition Engine: opt-in is selectable while stopped.
If the independently verified local ONNX model/runtime is missing, Standard/OpenCV
remains ACTIVE and the opt-in is pending. Restart or toggle off/on to recheck.
This is not an active AI model test until the verified model is available.
