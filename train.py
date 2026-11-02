import sys
from stable_baselines3 import SAC
from reach_env import ReachEnv

xml = sys.argv[1]                      # 実行時に arm_test.xml か arm7.xml を指定
env = ReachEnv(xml)
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("sac_" + xml.replace(".xml", ""))
print("学習が終わりました")
