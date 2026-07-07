"""
Equivalent of train_walk.py, but for the new Dict-observation pipeline.

This proves the new machinery (MultiInputPolicy, Dict obs, MultiDiscrete
actions including the craft slot) can actually transmit a learning signal -
NOT that walking right is a useful behavior. Same reward as the original
test (raw VelocityX) so the two runs are a fair comparison: if this doesn't
converge similarly to the original train_walk.py run, the bug is in the new
env/obs plumbing, not in RL fundamentals.

Run this AFTER bridge_smoke_test.py and the random-action check in the
README, and BEFORE spending real training time on the Combat or Explorer
gym reward functions.
"""

from stable_baselines3 import PPO
from terraria_hrl_env import TerrariaHRLEnv

env = TerrariaHRLEnv(reward_mode="smoke")

model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=0.0003)

print("Starting smoke-test training... switch to the Terraria window!")

# Same order of magnitude as the original walk test. If reward isn't
# trending up by the end of this, stop and debug the env before going
# any further - don't let this compound into a wasted Combat Gym run.
model.learn(total_timesteps=40000)

print("Smoke test finished. Saving model.")
model.save("hrl_pipeline_smoke_test")

print("Testing model...")
obs, _ = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    if done:
        obs, _ = env.reset()
