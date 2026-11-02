import sys
import mujoco
import imageio
from stable_baselines3 import SAC
from reach_env import ReachEnv

xml = sys.argv[1]
name = xml.replace(".xml", "")
env = ReachEnv(xml)
model = SAC.load("sac_" + name)

# カメラ：横から少し斜めに見る
renderer = mujoco.Renderer(env.model, height=480, width=640)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.4, 0, 0.3]
cam.distance = 2.0
cam.azimuth = 120
cam.elevation = -20

obs, _ = env.reset()
frames, hits = [], 0
for t in range(env.max_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, info = env.step(action)
    hits += int(info["hit"])
    renderer.update_scene(env.data, camera=cam)
    frames.append(renderer.render())

print(name, "最後の距離:", round(info["dist"], 3), "m / 塀に当たったステップ数:", hits)
imageio.mimsave(name + ".mp4", frames, fps=20)
print(name + ".mp4 を保存しました")
