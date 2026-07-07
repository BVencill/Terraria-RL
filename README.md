# Terraria RL Bridge — Setup & Capability Guide

## What this can actually do right now

Being precise about this matters more than it sounds, since it's easy to
assume "the code exists" means "the system works."

**Proven (you ran it, it worked):**
- Terraria ↔ Python TCP bridge, one direction of state out / actions in, every tick.
- A trivial single-behavior policy (walk right) trained end-to-end with PPO.

**Written, plausible, but NOT verified by me (no tModLoader to compile against):**
- Extended sensors: entity list, inventory features, biome flags, boss-milestone flags.
- The `Craft(ID)` cheat API.
- The `Dict`-observation env (`terraria_hrl_env.py`) that consumes all of the above.

**Not built yet — don't assume these exist:**
- Any actual Combat or Extraction reward function (both are stubs/TODOs).
- Any trained Combat, Extraction, or real Explorer model.
- `agent_manager.py` is not wired into a live loop yet — nothing currently calls it.
- No TEdit arenas exist yet for the gyms described in Phase 5.2/5.3.

So: you have a working nervous system and one working reflex. You don't yet
have a combat instinct, a mining instinct, or the brain that switches
between them. That's the honest state of things.

---

## Prerequisites

- Terraria + tModLoader (a dev/`tModLoader.dll` build capable of compiling mods)
- A C# environment tModLoader recognizes (VS Code with the C# extension is enough)
- Python 3.9+ with:
  ```bash
  pip install gymnasium stable-baselines3 numpy
  ```

---

## Step 1 — Install and compile the mod

1. Copy `RLStateSystem.cs` into your mod's source folder inside
   `Documents/My Games/Terraria/tModLoader/ModSources/<YourModName>/`.
2. In Terraria, open tModLoader → Mod Sources → **Build + Reload** on your mod.
3. If it fails to compile, the most likely culprits (flagged in code comments
   too): `ZoneHallow` may need to be `ZoneHoly` on older tModLoader/Terraria
   versions — check the compiler error and adjust.

## Step 2 — Sanity-check the raw bridge (do this before anything else)

Use the new `bridge_smoke_test.py` — no gym/SB3 dependency, so if something's
broken you're debugging the bridge, not your RL stack.

```bash
python bridge_smoke_test.py
```

Then load into a Terraria world with the mod enabled. It should print one
full `GameState` snapshot: position, health, grid size, occupied entity
slots, biome flags, inventory features, milestone flags. If `Entities`,
`Biome`, `Inventory`, or `Milestones` print as `MISSING`, the mod didn't
rebuild with the new fields — go back to Step 1.

To test the craft cheat: stand near enough raw materials for something
cheap (e.g. a Wood Sword needs Wood), run the script, and when prompted
enter that item's ItemID. You should see `LastCraftResult = 1`.

## Step 3 — Re-confirm the original walk task still works

This isn't a core function anymore, but it's your regression test — if this
breaks, something in the bridge fix regressed.

```bash
python train_walk.py
```

Should train without errors using the bug-fixed `terraria_env.py`
(same observation/action space as before, just the frame-drop and
stale-reset bugs fixed).

## Step 4 — Exercise the new Dict env with random actions

Before spending a training run on it, confirm `terraria_hrl_env.py` doesn't
throw shape errors:

```python
from terraria_hrl_env import TerrariaHRLEnv

env = TerrariaHRLEnv(reward_mode="raw")
obs, _ = env.reset()
for _ in range(200):
    obs, reward, done, _, info = env.step(env.action_space.sample())
    if done:
        obs, _ = env.reset()
print("OK - ran 200 random steps without error")
```

`reward_mode="raw"` only pays out the milestone bonus, so this is purely a
plumbing check, not a training run.

## Step 5 — Build a TEdit arena for the gym you're tackling first

This is a manual step, not code: use TEdit to build the Zombie-survival
arena (Combat Gym) or obstacle course (Explorer Gym) per the roadmap.
Nothing in Steps 1–4 depends on this, but training does.

## Step 6 — Implement the real reward function for that gym

`terraria_hrl_env._compute_reward()` currently has TODOs for both modes.
This is the piece I intentionally didn't guess at for you — say the word
and I'll build out whichever gym's reward function you pick first.

## Step 7 — Train the Worker, then repeat Steps 5–6 for the other gym

```python
from stable_baselines3 import PPO
from terraria_hrl_env import TerrariaHRLEnv

env = TerrariaHRLEnv(reward_mode="combat")  # or "explore"
model = PPO("MultiInputPolicy", env, verbose=1)  # MultiInputPolicy, not MlpPolicy - Dict obs space
model.learn(total_timesteps=100_000)
model.save("combat_worker")
```

## Step 8 — Wire `agent_manager.py` into a live loop

Right now it's a pure function (`select_mode(obs)`) that nothing calls. Once
you have at least two trained Workers, the loop becomes: each tick, ask
`AgentManager.select_mode(obs)` which mode to be in, call that mode's
`model.predict(obs)`, send the resulting action. Worth building this only
after Step 7 gives you a second Worker to switch between — one Worker has
nothing to switch to.

---

## Quick reference: what depends on what

```
RLStateSystem.cs  (must compile + connect first)
      │
      ├── bridge_smoke_test.py     (verifies the mod, no other deps)
      ├── terraria_env.py          (legacy scalar obs — walk_train.py)
      └── terraria_hrl_env.py      (Dict obs — Combat/Explorer gyms)
                │
                └── agent_manager.py   (needs 2+ trained Workers to matter)
```
