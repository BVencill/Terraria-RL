"""
The Manager / Switchboard (Phase 5.1).

This is deliberately a hand-written state machine, not a learned policy -
that's a real design choice, not a placeholder for laziness, so it's worth
naming the tradeoff: a hard-coded Manager is stable and debuggable (you can
always explain why it picked a Worker), at the cost of being exactly the
kind of hand-holding the project's "minimal hand-holding" goal wants to
avoid. Treat this as the "Crawl" stage of Crawl/Walk/Run for the Manager
specifically: once the Workers are stable, an obvious next step is replacing
these if/else rules with a small learned meta-controller (an options-style
policy over "which Worker to invoke" - see Sutton, Precup & Singh 1999;
Kulkarni et al. 2016 h-DQN; Vezhnevets et al. 2017 FeUdal Networks) that
takes the same coarse features below as its observation.

A middle ground worth considering before a full learned meta-controller:
keep these rules for safety-critical overrides (health critical -> flee,
no exceptions) and layer a bandit-style score on top for the genuinely
ambiguous cases (e.g. should an idle moment go to Combat-cleanup or
Extraction?) that tracks empirical outcome per (mode, context) and biases
choice toward what's worked, the way you'd track weapon performance.
"""

from enum import Enum


class Mode(Enum):
    EXPLORE = "explore"
    COMBAT = "combat"
    EXTRACT = "extract"


# Tune these against your own gyms - these are starting points, not measured values.
HEALTH_CRITICAL_FRAC = 0.25
THREAT_DETECT_RANGE = 400.0  # pixels; only look at entities within this range for a combat trigger


class AgentManager:
    def __init__(self):
        self.current_mode = Mode.EXPLORE
        self._forced_combat_until_clear = False

    def select_mode(self, obs: dict, event_flags: dict | None = None) -> Mode:
        """
        obs: the Dict observation produced by TerrariaHRLEnv._parse_obs().
        event_flags: optional dict of world-event booleans not carried in obs,
                     e.g. {"blood_moon": True, "invasion": True}. Wire these up
                     from GameState if/when you add them on the C# side.

        Priority order matters: safety-critical checks first, then event
        overrides, then normal task logic. Don't reorder without thinking
        about what happens when two conditions are true simultaneously.
        """
        event_flags = event_flags or {}

        # 1. Safety override - always wins, regardless of what else is happening.
        if obs["health_frac"][0] < HEALTH_CRITICAL_FRAC:
            return Mode.COMBAT  # closest thing to "fight or flee" until Workers differentiate the two

        # 2. Global event override, per the project's own "Rules of Engagement".
        if event_flags.get("blood_moon") or event_flags.get("invasion"):
            return Mode.COMBAT

        # 3. Reactive threat detection from the entity list.
        entities = obs["entities"]
        for e in entities:
            npc_type, rel_x, rel_y, life_frac, is_boss, damage = e
            if npc_type < 0:
                continue  # padding slot, no entity here
            dist = (rel_x ** 2 + rel_y ** 2) ** 0.5
            if dist < THREAT_DETECT_RANGE and (is_boss or damage > 0):
                return Mode.COMBAT

        # 4. Default task logic - placeholder. Extend with your own signal for
        #    "should I be mining right now" (e.g. inventory ore counts below a
        #    target threshold) vs. general exploration.
        return Mode.EXPLORE

    def get_worker_action_space_hint(self, mode: Mode):
        """
        Not enforced here - informational. Each Worker should be trained on
        its own gym with its own action space (per the roadmap: Explorer adds
        Grapple + Smart Cursor, Extraction forces Smart Cursor on). The
        Switchboard's job is choosing WHICH worker's model.predict() to call
        this tick, not reshaping the action space itself.
        """
        return {
            Mode.EXPLORE: "traversal/explorer worker action space",
            Mode.COMBAT: "combat worker action space",
            Mode.EXTRACT: "extraction worker action space (Smart Cursor forced on)",
        }[mode]
