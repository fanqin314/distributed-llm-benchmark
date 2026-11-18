import urllib.request, json, os, zipfile, time, sys

BASE = r"D:\workbuddy\Decentralized AI"
TAG = "b11932"
os.makedirs(os.path.join(BASE, "tools"), exist_ok=True)
os.makedirs(os.path.join(BASE, "models"), exist_ok=True)

def log(m):
    print(m, flush=True)

def dl(url, dest, label, min_size=1024 * 1024):
    if os.path.exists(dest) and os.path.getsize(dest) > min_size:
        log(f"[跳过] {label} 已完整 {os.path.getsize(dest)/1024/1024:.0f}MB")
        return True
    tmp = dest + ".part"
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        last = 0
        while True:
            b = r.read(256 * 1024)
            if not b:
                break
            f.write(b)
            got += len(b)
            if time.time() - last > 8:
                sp = got / (time.time() - t0) / 1024 / 1024
                log(f"[{label}] {got/1024/1024:.0f}/{total/1024/1024:.0f}MB  {sp:.2f}MB/s")
                last = time.time()
    os.replace(tmp, dest)
    log(f"[完成] {label} {os.path.getsize(dest)/1024/1024:.0f}MB 用时 {time.time()-t0:.0f}s")
    return True

log("=== llama.cpp 二进制 ===")
try:
    req = urllib.request.Request(
        f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{TAG}",
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"},
    )
    rel = json.loads(urllib.request.urlopen(req, timeout=25).read())
    target = None
    for a in rel.get("assets", []):
        n = a["name"]
        if "win-cpu-x64" in n and n.endswith(".zip"):
            target = (n, a["browser_download_url"], a["size"])
            break
    if not target:
        log("!! 没找到 win-cpu-x64 资产")
        for a in rel.get("assets", []):
            log("    " + a["name"])
    else:
        log(f"目标: {target[0]} ({target[2]/1024/1024:.0f}MB)")
        z = os.path.join(BASE, "tools", target[0])
        dl(target[1], z, "llama.cpp")
        out = os.path.join(BASE, "tools", "llama")
        os.makedirs(out, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
        log(f"[解压] -> {out}")
        for root, dirs, files in os.walk(out):
            for fn in sorted(files):
                if fn.endswith(".exe"):
                    log("    " + os.path.relpath(os.path.join(root, fn), out))
except Exception as e:
    log(f"!! 失败 {type(e).__name__}: {e}")

log("=== 模型 ===")
try:
    u = (
        "https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/repo"
        "?Revision=master&FilePath=qwen2.5-1.5b-instruct-q4_k_m.gguf"
    )
    dest = os.path.join(BASE, "models", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    dl(u, dest, "Qwen2.5-1.5B-Q4", 900 * 1024 * 1024)
except Exception as e:
    log(f"!! 失败 {type(e).__name__}: {e}")

log("=== DONE ===")
