"""Pure post-Shift scheduler for the two Blood Awakening raids.

One observation / one action at most per tick; game input is owned by main.
Heatwave's two observed cycles are kept separate from the activation attempt.
"""
from dataclasses import dataclass


@dataclass
class StableIcon:
    value: bool = True
    candidate: bool | None = None
    since: float = 0.0
    frames: int = 0

    def invalidate(self):
        self.candidate = None
        self.frames = 0

    def observe(self, value, now, stable_for, min_frames):
        if self.candidate != value:
            self.candidate, self.since, self.frames = value, now, 1
        else:
            self.frames += 1
        if (value != self.value and self.frames >= min_frames
                and now - self.since >= stable_for):
            self.value = value
            return value
        return None


class BloodSkillScheduler:
    """Preserves independent timers and gives Heatwave bounded activation priority.

    ``seismic`` may be a verified ready/not-ready observation, or None when
    no Seismic Shock template is available. False means no positive ready
    evidence, not verified cooldown. None preserves the old skill-5 timer.
    """
    priority = ('4', '5', '3', '1')

    def __init__(self, intervals, now, stable_for=.35, min_frames=3,
                 retry_interval=.5, action_gap=.12):
        self.intervals = {slot: float(intervals[slot]) for slot in ('5', '3', '1')}
        self.ready_at = dict.fromkeys(self.intervals, now)
        self.stable_for = float(stable_for)
        self.min_frames = int(min_frames)
        self.retry_interval = float(retry_interval)
        self.action_gap = float(action_gap)
        self.heat = StableIcon()
        self.seismic = StableIcon(value=False)
        self.seismic_visual = False
        self.seismic_pending = False
        self.seismic_retry_at = now
        self.seismic_sent_at = None
        self.stage = 'activation_pending'
        self.attempted = False
        self.heat_attempts = 0
        self.seismic_attempts = 0
        self.retry_at = now
        self.next_action = now
        self.confirmed_activations = 0
        self.completed_cycles = 0
        self.last_event = None
        self.last_hud_at = None
        self.require_fresh_heat = False
        self.last_heat_at = None
        self.interrupted_heat_phase = None
        self.max_observation_gap = max(2.0, 3 * self.stable_for)

    def interrupt_heat(self):
        self.heat.invalidate()
        self.require_fresh_heat = True
        if self.stage == 'activation_pending' and self.attempted:
            self.attempted = False
            self.heat_attempts = 0
            self.interrupted_heat_phase = 0
            self.last_event = 'Heatwave attempt interrupted; toggle held until two observed cycles'
        elif self.interrupted_heat_phase is not None:
            self.interrupted_heat_phase = 0

    def observe(self, now, *, hud, heatwave=False, seismic=None):
        self.last_event = None
        # Never bridge a key attempt across an unobserved interval. An icon
        # absent after a cutscene/pause is not evidence of the old key's effect.
        gap = (self.last_hud_at is not None and
               now - self.last_hud_at >= self.max_observation_gap)
        if not hud or gap:
            self.interrupt_heat()
            self.seismic.invalidate()
            self.last_hud_at = None
            self.last_heat_at = None
            if not hud:
                return None
        self.last_hud_at = now
        self.seismic_visual = seismic is not None
        if self.seismic_visual:
            self.seismic.observe(bool(seismic), now, self.stable_for, self.min_frames)
            if (self.seismic_pending and self.seismic.candidate is False
                    and self.seismic.frames >= self.min_frames
                    and now - self.seismic.since >= self.stable_for):
                self.seismic_pending = False
                self.seismic_attempts = 0
                self.seismic_retry_at = now
                self.last_event = 'Seismic Shock ready icon no longer observed; cooldown unverified'
        else:
            self.seismic.invalidate()
            self.seismic.value = False
            self.seismic_pending = False
        if heatwave is None:
            if self.interrupted_heat_phase is not None:
                self.interrupted_heat_phase = 0
            if (self.last_heat_at is not None and
                    now - self.last_heat_at >= self.max_observation_gap):
                self.interrupt_heat()
            self.heat.invalidate()
            self.require_fresh_heat = True
            edge = None
        else:
            if (self.last_heat_at is not None and
                    now - self.last_heat_at >= self.max_observation_gap):
                self.interrupt_heat()
            self.last_heat_at = now
            edge = self.heat.observe(bool(heatwave), now, self.stable_for, self.min_frames)
        if (self.require_fresh_heat and self.heat.frames >= self.min_frames
                and now - self.heat.since >= self.stable_for):
            self.require_fresh_heat = False
        if self.interrupted_heat_phase is not None:
            # A key may have succeeded before HUD loss. Require a complete
            # dark/bright/dark/bright sequence; never guess from elapsed time.
            phase = self.interrupted_heat_phase
            stable = (heatwave is not None and self.heat.frames >= self.min_frames
                      and now - self.heat.since >= self.stable_for)
            if stable and bool(heatwave) == (phase % 2 == 1):
                self.interrupted_heat_phase += 1
                if self.interrupted_heat_phase == 4:
                    self.interrupted_heat_phase = None
                    self.completed_cycles = 2
                    self.retry_at = now
                    self.last_event = 'Heatwave resynchronized after two observed cycles'
                    return '4' if now >= self.next_action else None
        elif self.stage == 'activation_pending':
            # Do not count a key press as an activation. Confirm a stable
            # hidden icon while HUD is observable, following an actual attempt.
            if self.attempted and edge is False:
                self.stage = 'first_visible'
                self.completed_cycles = 0
                self.confirmed_activations += 1
                self.attempted = False
                self.heat_attempts = 0
                self.last_event = 'Heatwave activation CONFIRMED; first cooldown started'
            else:
                if (heatwave and self.heat.value and now >= self.retry_at
                        and now >= self.next_action
                        and not self.require_fresh_heat):
                    return '4'
                # A genuinely unobserved ready icon must not starve every
                # other skill indefinitely. Wait out the debounce, but retain
                # priority immediately when Heatwave becomes visible again.
                # Give an initial key and one prompt retry exclusive priority.
                # Further retries are paced, and other skills can run between them.
                # Unknown observations never hold the entire scheduler hostage.
                if (heatwave is not None and self.heat_attempts < 2 and
                        (heatwave or self.attempted or self.heat.value
                         or self.heat.frames < self.min_frames
                         or now - self.heat.since < self.stable_for)):
                    return None
        elif self.stage == 'first_visible' and edge is True:
            self.completed_cycles = 1
            self.stage = 'second_hidden'
            self.last_event = 'Heatwave first cooldown ended; do not press Skill 4'
        elif self.stage == 'second_hidden' and edge is False:
            self.stage = 'true_ready'
            self.last_event = 'Heatwave second cooldown started'
        elif self.stage == 'true_ready' and edge is True:
            self.completed_cycles = 2
            self.stage = 'activation_pending'
            self.attempted = False
            self.retry_at = now
            self.heat_attempts = 0
            self.last_event = 'Heatwave second cooldown ended; awaiting activation confirmation'
            return '4' if now >= self.next_action else None
        if now < self.next_action:
            return None
        # Don't start a lower animation while a possible higher-priority
        # ready edge is being debounced.
        if self.stage == 'true_ready' and heatwave:
            return None
        if self.seismic_visual:
            if seismic:
                if (self.seismic.value and self.seismic.frames >= self.min_frames
                        and now - self.seismic.since >= self.stable_for
                        and now >= self.seismic_retry_at):
                    return '5'
                if self.seismic_attempts < 2:
                    return None
            if (not seismic and self.seismic_pending and self.seismic_sent_at is not None
                    and now - self.seismic_sent_at < max(.5, 2*self.stable_for)):
                return None  # brief confirmation grace; never block indefinitely
        elif now >= self.ready_at['5']:
            return '5'
        for slot in ('3', '1'):
            if now >= self.ready_at[slot]:
                return slot
        return None

    def sent(self, slot, now):
        """Record a sent key only; never advance Heatwave cooldown here."""
        self.next_action = now + self.action_gap
        if slot == '4':
            self.attempted = True
            self.heat_attempts += 1
            self.retry_at = now + (self.retry_interval if self.heat_attempts < 2 else max(1.5, self.retry_interval))
        else:
            self.ready_at[slot] = now + self.intervals[slot]
            if slot == '5' and self.seismic_visual:
                self.seismic_pending = True
                self.seismic_sent_at = now
                self.seismic_attempts += 1
                self.seismic_retry_at = now + (self.retry_interval if self.seismic_attempts < 2 else max(1.5, self.retry_interval))
