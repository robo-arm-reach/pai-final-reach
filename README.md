# pai-final-reach
# 障害物越しのリーチ：6軸アームと7軸アームの比較

塀の向こうにある目標へ手を伸ばすタスクを、MuJoCo上の6軸アームと7軸アームに強化学習（SAC）で学習させ、7軸目（冗長な自由度）の効果を比較しました。

## ファイル
- `arm_test.xml`：6軸アームと環境（塀・目標）のモデル
- `arm7.xml`：7軸アーム（6軸に肩ロールを追加）のモデル
- `reach_env.py`：Gymnasium環境（観測・行動・報酬）
- `train.py`：Stable-Baselines3のSACで学習
- `evaluate.py`：学習した方策を動かし、最後の距離・塀への接触数を表示して動画を保存

## 実行方法
pip install mujoco gymnasium stable-baselines3 "imageio[ffmpeg]"
python train.py arm7.xml
python evaluate.py arm7.xml

## 結果（各条件1回）
| 腕 | 学習量 | 最後の距離 | 塀への接触 |
|---|---|---|---|
| 6軸 | 3万ステップ | 0.106 m | 19 ステップ |
| 7軸 | 3万ステップ | 0.339 m | 2 ステップ |
| 6軸 | 10万ステップ | 0.109 m | 0 ステップ |
| 7軸 | 10万ステップ | 0.080 m | 0 ステップ |
