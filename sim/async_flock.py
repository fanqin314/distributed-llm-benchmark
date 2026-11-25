"""
同协异步 · 船阵仿真（异步版 boids）

和经典 boids 的唯一区别：每一步随机打乱更新顺序，每艘船只看得见最近的 k 个邻居。
「异步」和「局部」这两件事，就是"同协异步"的全部。

依赖：numpy（无需 Mesa；说明见 README）

用法：
    python async_flock.py --demo      # 单次实验，打印 φ 随时间变化
    python async_flock.py --sweep     # 全量扫描 → results.csv / summary.md / figure.png
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np


class Flock:
    """环形平面上的 N 个个体：异步更新、局部观察、允许偷懒与掉线。"""

    def __init__(self, n=40, k=3, lazy_frac=0.0, seed=0, world=1.0, speed=0.010):
        self.rng = np.random.default_rng(seed)
        self.n, self.k, self.world, self.speed = n, k, world, speed

        self.pos = self.rng.random((n, 2)) * world
        ang = self.rng.random(n) * 2 * np.pi
        self.vel = np.stack([np.cos(ang), np.sin(ang)], axis=1) * speed

        self.alive = np.ones(n, bool)
        # 偷懒节点：在场、能被看见，但从不更新自己的方向
        self.lazy = self.rng.random(n) < lazy_frac

    # ---------- 观察：最近的 k 个活着的邻居 ----------
    def _neighbors(self, i):
        d = self.pos - self.pos[i]
        d -= np.round(d / self.world) * self.world      # 环形世界的最短位移
        dist = np.linalg.norm(d, axis=1)
        dist[i] = np.inf
        dist[~self.alive] = np.inf
        k = min(self.k, int(self.alive.sum()) - 1)
        idx = np.argpartition(dist, k - 1)[:k]
        return idx, d[idx], dist[idx]

    # ---------- 单个体更新：立刻生效（高斯-赛德尔），不等别人 ----------
    def _update(self, i):
        idx, d, dist = self._neighbors(i)
        unit = d / np.maximum(dist[:, None], 1e-9)

        v_hat = self.vel[idx] / np.maximum(np.linalg.norm(self.vel[idx], axis=1, keepdims=True), 1e-9)
        desired = (
            1.00 * v_hat.mean(axis=0)                                        # 对齐
            + 0.20 * (-unit.mean(axis=0))                                    # 聚合：朝邻居质心
            + 0.25 * (-unit / np.maximum(dist[:, None], 0.02)).mean(axis=0)  # 分离：别撞
        )
        desired /= max(np.linalg.norm(desired), 1e-9)
        desired += self.rng.normal(0, 0.03, 2)
        desired /= max(np.linalg.norm(desired), 1e-9)

        v = 0.85 * self.vel[i] / max(np.linalg.norm(self.vel[i]), 1e-9) + 0.15 * desired
        v /= max(np.linalg.norm(v), 1e-9)
        self.vel[i] = v * self.speed
        self.pos[i] = (self.pos[i] + self.vel[i]) % self.world

    def step(self):
        """异步：每一步随机决定谁先动。"""
        for i in self.rng.permutation(self.n):
            if self.alive[i] and not self.lazy[i]:
                self._update(i)

    # ---------- 指标 ----------
    def order_parameter(self):
        """φ ∈ [0,1]，1 表示全体同向。"""
        v = self.vel / np.maximum(np.linalg.norm(self.vel, axis=1, keepdims=True), 1e-9)
        return float(np.linalg.norm(v[self.alive].mean(axis=0)))

    def dropout(self, frac=0.2):
        """随机让一部分个体彻底掉线（不算邻居）。"""
        idx = np.where(self.rng.random(self.n) < frac)[0]
        idx = idx[: max(0, int(self.alive.sum()) - 3)]
        self.alive[idx] = False
        return len(idx)


def run_once(n, k, lazy_frac, seed, max_steps=400, form_phi=0.90, hold=10):
    f = Flock(n=n, k=k, lazy_frac=lazy_frac, seed=seed)
    formed_at, ok, phi = None, 0, 0.0
    for t in range(max_steps):
        f.step()
        phi = f.order_parameter()
        if phi >= form_phi:
            ok += 1
            if formed_at is None and ok >= hold:
                formed_at = t - hold + 1
        else:
            ok = 0

    recovery, collapsed = None, None
    if formed_at is not None:
        # 同样的初始条件重跑到成队，然后踢掉 20% 节点，看多久恢复
        g = Flock(n=n, k=k, lazy_frac=lazy_frac, seed=seed)
        for _ in range(formed_at + hold):
            g.step()
        g.dropout(0.2)
        for t in range(200):
            g.step()
            if g.order_parameter() >= form_phi:
                recovery = t + 1
                break
        collapsed = recovery is None

    return {
        "n": n, "k": k, "lazy": lazy_frac, "seed": seed,
        "formed": formed_at is not None,
        "steps_to_form": formed_at if formed_at is not None else -1,
        "phi_final": round(phi, 4),
        "collapsed_after_dropout": collapsed,
        "recovery_steps": recovery,
    }


def sweep(outdir, n=40, seeds=25, max_steps=400,
          ks=(2, 3, 5, 8), lazies=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5)):
    rows, t0, total = [], time.time(), len(ks) * len(lazies) * seeds
    for k in ks:
        for lazy in lazies:
            for s in range(seeds):
                rows.append(run_once(n=n, k=k, lazy_frac=lazy, seed=s, max_steps=max_steps))
                if len(rows) % 50 == 0 or len(rows) == total:
                    print(f"  {len(rows)}/{total}  {time.time()-t0:.1f}s", flush=True)

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "results.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for k in ks:
        for lazy in lazies:
            sub = [r for r in rows if r["k"] == k and r["lazy"] == lazy]
            formed = [r for r in sub if r["formed"]]
            rate = len(formed) / len(sub)
            steps = float(np.median([r["steps_to_form"] for r in formed])) if formed else None
            collapsed = [r for r in formed if r["collapsed_after_dropout"]]
            c_rate = len(collapsed) / len(formed) if formed else None
            rec = [r["recovery_steps"] for r in formed if r["recovery_steps"] is not None]
            summary.append({
                "k": k, "lazy": lazy,
                "formation_rate": round(rate, 3),
                "median_steps": round(steps, 1) if steps is not None else None,
                "collapse_rate": round(c_rate, 3) if c_rate is not None else None,
                "median_recovery": round(float(np.median(rec)), 1) if rec else None,
            })

    md = ["# 实验结果（自动生成，勿手改）", "",
          f"- 个体数 N = {n}；每个格子 {seeds} 个随机种子；单次上限 {max_steps} 步",
          "- 成队判据：φ ≥ 0.90 连续 10 步（φ 为全体方向一致性，1 = 完全同向）",
          "- 崩溃判据：成队后随机踢掉 20% 节点，200 步内回不到 φ ≥ 0.90",
          "- 偷懒节点：在场、能被看见，但从不更新方向", "",
          "| k | 偷懒比例 | 成队率 | 成队步数(中位) | 踢人后崩溃率 | 恢复步数(中位) |",
          "|---|---|---|---|---|---|"]
    for r in summary:
        md.append("| {} | {:.0%} | {:.0%} | {} | {} | {} |".format(
            r["k"], r["lazy"], r["formation_rate"],
            r["median_steps"] if r["median_steps"] is not None else "—",
            f"{r['collapse_rate']:.0%}" if r["collapse_rate"] is not None else "—",
            r["median_recovery"] if r["median_recovery"] is not None else "—"))
    with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    return summary, time.time() - t0


def figure(summary, ks, lazies, path):
    from PIL import Image, ImageDraw

    W, H = 960, 620
    L, R, T, B = 90, 40, 70, 540
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    xs = [L + (W - R - L) * i / (len(lazies) - 1) for i in range(len(lazies))]
    ys = lambda v: B + (T - B) * v

    d.line([(L, B), (W - R, B)], fill=(40, 40, 40), width=2)
    d.line([(L, B), (L, T)], fill=(40, 40, 40), width=2)
    for v in (0, .25, .5, .75, 1.0):
        d.line([(L, ys(v)), (W - R, ys(v))], fill=(228, 228, 228), width=1)
        d.text((L - 38, ys(v) - 5), f"{int(v*100)}%", fill=(70, 70, 70))
    for x, lz in zip(xs, lazies):
        d.text((x - 14, B + 14), f"{int(lz*100)}%", fill=(70, 70, 70))

    d.text((W // 2 - 100, H - 40), "fraction of lazy nodes", fill=(40, 40, 40))
    d.text((20, T - 40), "formation rate", fill=(40, 40, 40))
    d.text((L, 22), "Async flocking: formation rate vs lazy nodes (N=40, 25 seeds)", fill=(20, 20, 20))

    colors = {2: (198, 62, 42), 3: (32, 74, 132), 5: (38, 128, 88), 8: (142, 108, 40)}
    for k in ks:
        pts = [(x, ys(next(s for s in summary if s["k"] == k and s["lazy"] == lz)["formation_rate"]))
               for x, lz in zip(xs, lazies)]
        d.line(pts, fill=colors[k], width=3)
        for p in pts:
            d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=colors[k])
        lx, ly = W - R - 160, T + 14 + 26 * list(ks).index(k)
        d.line([(lx, ly), (lx + 26, ly)], fill=colors[k], width=3)
        d.text((lx + 34, ly - 5), f"k = {k}", fill=(40, 40, 40))

    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=25)
    args = ap.parse_args()

    if args.demo:
        f = Flock(n=args.n, k=3, lazy_frac=0.0, seed=1)
        for t in range(0, 130, 10):
            print(f"t={t:3d}  phi={f.order_parameter():.3f}")
            for _ in range(10):
                f.step()
        return

    ks, lazies = (2, 3, 5, 8), (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
    summary, secs = sweep(args.out, n=args.n, seeds=args.seeds, ks=ks, lazies=lazies)
    figure(summary, ks, lazies, os.path.join(args.out, "figure.png"))
    print(f"done in {secs:.1f}s → {args.out}")
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
