import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import json
from collections import deque

class TerrariaEnv(gym.Env):
    """
    Same observation/action space as the original - this is a drop-in
    replacement for train_walk.py. The changes are correctness fixes only:

    1. _get_state() used to silently DISCARD complete state lines whenever
       more than one had arrived since the last recv() - it kept only the
       most recent one. If Python ever falls a tick behind Terraria (which
       will happen), that meant lost reward and an action count that no
       longer matched the tick count. Fixed with a FIFO queue: every state
       line that arrives gets consumed exactly once, in order.
    2. reset() didn't clear stale buffered data, so it could return a state
       from just before the teleport actually applied. Fixed by clearing
       the queue and discarding a couple of settle frames after teleporting.
    3. TCP_NODELAY wasn't set, adding latency to every small packet.
    """

    def __init__(self):
        super(TerrariaEnv, self).__init__()

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        self.host = '127.0.0.1'
        self.port = 7778
        self.conn = None
        self.socket = None
        self.buffer = ""
        self._state_queue = deque()

        self.frame_skip = 4
        self._settle_frames = 2  # frames to discard after a reset, to let the teleport/velocity settle

        self._start_server()

    def _start_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        print("Waiting for Terraria connection...")
        self.conn, addr = self.socket.accept()
        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Terraria Connected: {addr}")

    def _fill_queue(self):
        """Reads whatever is available and appends every complete line found
        to _state_queue, in order. Never discards a line."""
        data = self.conn.recv(4096)
        if not data:
            raise ConnectionError("Game closed")
        self.buffer += data.decode('utf-8')

        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            self.buffer = lines.pop()  # last element is either "" or an incomplete line - keep it
            for line in lines:
                if line.strip():
                    self._state_queue.append(json.loads(line))

    def _get_state(self):
        """Returns the OLDEST unconsumed state (FIFO) - never skips a tick."""
        while not self._state_queue:
            self._fill_queue()
        return self._state_queue.popleft()

    def _send_action(self, action):
        move_val = 0
        if action == 1: move_val = -1
        if action == 2: move_val = 1

        command = {
            "Move": move_val,
            "Jump": False,
            "UseItem": False,
            "Reset": False
        }
        self.conn.sendall((json.dumps(command) + "\n").encode('utf-8'))

    def step(self, action):
        total_reward = 0.0
        done = False
        game_state = None

        for _ in range(self.frame_skip):
            self._send_action(action)
            game_state = self._get_state()

            reward = game_state['VelocityX']
            total_reward += reward

            if game_state['Health'] <= 0:
                done = True
                total_reward -= 10
                break

        obs = np.array([game_state['VelocityX']], dtype=np.float32)
        return obs, total_reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Drop anything left over from the previous episode - otherwise the
        # first read below could return a pre-reset, stale position/velocity.
        self._state_queue.clear()
        self.buffer = ""

        command = {"Move": 0, "Jump": False, "UseItem": False, "Reset": True}
        self.conn.sendall((json.dumps(command) + "\n").encode('utf-8'))

        # The teleport is applied on the tick after this command is received,
        # so the very next line or two can still reflect pre-teleport motion.
        # Discard a couple of frames to let position/velocity settle before
        # trusting the observation. This is a heuristic, not a guarantee - if
        # you need exact synchronization, add an explicit "IsPostReset" flag
        # to GameState on the C# side and loop until you see it.
        game_state = None
        for _ in range(self._settle_frames):
            game_state = self._get_state()

        obs = np.array([game_state['VelocityX']], dtype=np.float32)
        return obs, {}

    def close(self):
        if self.conn: self.conn.close()
        if self.socket: self.socket.close()
