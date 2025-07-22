import numpy as np, matplotlib.pyplot as plt
import matplotlib.animation as anim, sys, argparse, pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--npy_path", default="/home/challenge-user/challenge-audio-to-gesture/DiffuseStyleGesture/sarper_lmdb/inference_result/20250714_232754_smoothing_SG_minibatch_3060_[0, 1, 0, 0, 0, 0]_123456.npy",help="path input")
parser.add_argument("--out", default="/home/challenge-user/challenge-audio-to-gesture/DiffuseStyleGesture/sarper_lmdb/inference_result/output_video2.mp4", help="output mp4 (default = same name)")
args = parser.parse_args()

poses = np.load(args.npy_path)            # (T, 63) OR (T, 9, 7)
if poses.ndim == 2:
    poses = poses.reshape(-1, 9, 7)

T = poses.shape[0]
out_mp4 = args.out or pathlib.Path(args.npy_path).with_suffix(".mp4")

# joint connections for a simple stick-figure (adjust if you want)
edges = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8)]

fig = plt.figure(figsize=(4,6))
ax  = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([0,2])
lines = [ax.plot([],[],[], lw=3, c='k')[0] for _ in edges]

def update(frame):
    pts = poses[frame,:,:3]               # (9,3) xyz
    for ln,(i,j) in zip(lines, edges):
        ln.set_data  ([pts[i,0], pts[j,0]],[pts[i,2], pts[j,2]])
        ln.set_3d_properties([pts[i,1], pts[j,1]])
    ax.set_title(f"frame {frame}/{T}")
    return lines

step = 10                       # change 10→5→3 to taste
poses = poses[::step]           # down-sample T dimension
T     = poses.shape[0]
ani   = anim.FuncAnimation(fig, update, frames=T, interval=100, blit=True)

#ani = anim.FuncAnimation(fig, update, frames=T, interval=33, blit=True)
ani.save(out_mp4, fps=10, bitrate=3200, codec='mpeg4')
print("Saved", out_mp4)
