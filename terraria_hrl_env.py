"""
Phase 5 environment: the full "God Mode" sensor state (grid + entities +
inventory + biome + milestones) and the Craft(ID) cheat action, on top of
the same TCP bridge as terraria_env.py.

This is a SEPARATE file from terraria_env.py on purpose: train_walk.py and
the Phase 4 result depend on the old scalar observation space, and this
Dict space is not a drop-in replacement for it. Point new Worker training
scripts (Combat Gym, Explorer Gym) at this file instead.

Requires SB3's MultiInputPolicy (not MlpPolicy) since the observation space
is a Dict:
    from stable_baselines3 import PPO
    model = PPO("MultiInputPolicy", env, verbose=1)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import json
from collections import deque

GRID_RADIUS = 20
GRID_SIZE = GRID_RADIUS * 2 + 1  # 41
MAX_ENTITIES = 8
ENTITY_FEATURES = 6   # NpcType, RelX, RelY, LifeFrac, IsBoss, Damage
INVENTORY_FEATURES = 7
BIOME_FEATURES = 16
MILESTONE_FEATURES = 11


class TerrariaHRLEnv(gym.Env):
    def __init__(self, reward_mode="raw", craftable_item_ids=None, host="127.0.0.1", port=7778):
        """
        reward_mode: "raw" (no shaping, just exposes state - use for debugging),
                     "combat", or "explore". See _compute_reward().
        craftable_item_ids: list of ItemIDs the Craft action space can choose from.
                     Index 0 is reserved for "craft nothing".
        """
        super().__init__()
        self.reward_mode = reward_mode
        self.craftable_item_ids = craftable_item_ids or []
        self._craft_request_counter = 0

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=3, shape=(GRID_SIZE, GRID_SIZE), dtype=np.float32),
            "entities": spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_ENTITIES, ENTITY_FEATURES), dtype=np.float32),
            "inventory": spaces.Box(low=0, high=np.inf, shape=(INVENTORY_FEATURES,), dtype=np.float32),
            "biome": spaces.MultiBinary(BIOME_FEATURES),
            "milestones": spaces.MultiBinary(MILESTONE_FEATURES),
            "health_frac": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        })

        # Move: 0=idle,1=left,2=right | Jump: 0/1 | UseItem: 0/1 | Craft: index into craftable_item_ids (0 = none)
        self.action_space = spaces.MultiDiscrete([3, 2, 2, len(self.craftable_item_ids) + 1])

        self.host, self.port = host, port
        self.conn = None
        self.socket = None
        self.buffer = ""
        self._state_queue = deque()
        self._prev_milestones = None
        self._visited_cells = set()  # crude count-based novelty for the explore reward - see TODO below

        self.frame_skip = 4
        self._settle_frames = 2

        self._start_server()

    # --- networking (same pattern as terraria_env.py) ---
    def _start_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        print("Waiting for Terraria connection...")
        self.conn, addr = self.socket.accept()
        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Terraria Connected: {addr}")

    def _fill_queue(self):
        data = self.conn.recv(65536)  # bigger buffer - the grid array alone is ~1681 ints of JSON
        if not data:
            raise ConnectionError("Game closed")
        self.buffer += data.decode('utf-8')
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            self.buffer = lines.pop()
            for line in lines:
                if line.strip():
                    self._state_queue.append(json.loads(line))

    def _get_raw_state(self):
        while not self._state_queue:
            self._fill_queue()
        return self._state_queue.popleft()

    def _send_action(self, move, jump, use_item, craft_item_id=0, craft_request_id=0):
        command = {
            "Move": move, "Jump": bool(jump), "UseItem": bool(use_item), "Reset": False,
            "CraftItemID": craft_item_id, "CraftRequestId": craft_request_id,
        }
        self.conn.sendall((json.dumps(command) + "\n").encode('utf-8'))

    # --- state parsing ---
    def _parse_obs(self, raw):
        grid = np.array(raw["Grid"], dtype=np.float32).reshape(GRID_SIZE, GRID_SIZE)

        entities = np.zeros((MAX_ENTITIES, ENTITY_FEATURES), dtype=np.float32)
        for i, e in enumerate(raw["Entities"][:MAX_ENTITIES]):
            entities[i] = [e["NpcType"], e["RelX"], e["RelY"], e["LifeFrac"], float(e["IsBoss"]), e["Damage"]]

        return {
            "grid": grid,
            "entities": entities,
            "inventory": np.array(raw["Inventory"], dtype=np.float32),
            "biome": np.array(raw["Biome"], dtype=np.int8),
            "milestones": np.array(raw["Milestones"], dtype=np.int8),
            "health_frac": np.array([raw["Health"] / max(raw["MaxHealth"], 1)], dtype=np.float32),
            "velocity": np.array([raw["VelocityX"], raw["VelocityY"]], dtype=np.float32),
        }

    # --- reward ---
    # NOTE: this is deliberately generic. The real Combat Gym reward
    # ("Efficiency (Time) + Damage Dealt - Damage Taken", per the roadmap) and
    # Explorer Gym reward ("Fog of War clearing") are Phase 5.2/5.3 work, not
    # implemented yet. What's here is (a) a milestone bonus that fires off the
    # boss-downed flags for free, shared by every mode, and (b) a placeholder
    # count-based novelty term for "explore" so the plumbing exists to build on.
    #
    # Two things worth deciding before you flesh these out further (see the
    # accompanying chat message): normalize combat reward against equipped
    # gear so early- and late-game fights are comparable, and replace the
    # count-based novelty bonus with RND (Burda et al. 2018) once the tabular
    # version starts saturating - visited-cell sets don't scale past a fairly
    # small explored area.
    def _milestone_bonus(self, milestones):
        bonus = 0.0
        if self._prev_milestones is not None:
            newly_downed = np.logical_and(milestones == 1, self._prev_milestones == 0)
            bonus = 100.0 * np.sum(newly_downed)
        self._prev_milestones = milestones.copy()
        return bonus

    def _explore_bonus(self, raw):
        cell = (round(raw["PlayerX"] / 16 / 4), round(raw["PlayerY"] / 16 / 4))  # coarse 4-tile buckets
        if cell not in self._visited_cells:
            self._visited_cells.add(cell)
            return 1.0
        return 0.0

    def _compute_reward(self, raw, obs, done):
        reward = self._milestone_bonus(obs["milestones"])

        if self.reward_mode == "combat":
            reward += -0.01  # small per-tick time penalty -> encourages faster kills
            # TODO: += damage dealt to tracked boss this tick, -= damage taken this tick
        elif self.reward_mode == "explore":
            reward += self._explore_bonus(raw)
        elif self.reward_mode == "smoke":
            # Deliberately NOT a useful behavior - this exists only to prove
            # MultiInputPolicy + the Dict/MultiDiscrete pipeline can transmit a
            # learning signal at all, using the exact same dense reward as the
            # original train_walk.py test for a like-for-like comparison.
            reward += raw["VelocityX"]
        # "raw" mode: milestone bonus only, for debugging the pipeline itself

        if raw["Health"] <= 0:
            reward -= 10.0

        return float(reward)

    # --- gym API ---
    def step(self, action):
        move_idx, jump, use_item, craft_idx = action
        move = {0: 0, 1: -1, 2: 1}[int(move_idx)]

        craft_item_id, craft_request_id = 0, 0
        if craft_idx > 0:
            craft_item_id = self.craftable_item_ids[craft_idx - 1]
            self._craft_request_counter += 1
            craft_request_id = self._craft_request_counter

        raw = None
        for _ in range(self.frame_skip):
            self._send_action(move, jump, use_item, craft_item_id, craft_request_id)
            raw = self._get_raw_state()
            craft_item_id, craft_request_id = 0, 0  # only fire the craft attempt on the first sub-frame
            if raw["Health"] <= 0:
                break

        obs = self._parse_obs(raw)
        done = raw["Health"] <= 0
        reward = self._compute_reward(raw, obs, done)
        info = {"craft_result": raw.get("LastCraftResult", 0)}
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state_queue.clear()
        self.buffer = ""
        self._prev_milestones = None

        command = {"Move": 0, "Jump": False, "UseItem": False, "Reset": True,
                   "CraftItemID": 0, "CraftRequestId": 0}
        self.conn.sendall((json.dumps(command) + "\n").encode('utf-8'))

        raw = None
        for _ in range(self._settle_frames):
            raw = self._get_raw_state()

        return self._parse_obs(raw), {}

    def close(self):
        if self.conn: self.conn.close()
        if self.socket: self.socket.close()
