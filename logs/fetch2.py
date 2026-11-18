import urllib.request, os, time, zipfile

BASE = r"D:\workbuddy\Decentralized AI"


def log(m):
    print(m, flush=True)


def dl(url, dest, label):
    if os.path.exists(dest) and os.path.getsize(dest) > 1024 * 1024:
        log(f"[skip] {label} {os.path.getsize(dest)/1024/1024:.0f}MB")
        return dest
    tmp = dest + ".part"
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
        got = 0
        while True:
            b = r.read(256 * 1024)
            if not b:
                break
            f.write(b)
            got += len(b)
    log(f"[done] {label} {got/1024/1024:.0f}MB in {time.time()-t0:.0f}s")
    os.replace(tmp, dest)
    return dest


log("=== 1/3 llama.cpp b11429 Windows ===")
z = os.path.join(BASE, "tools", "llama-b11429-bin-win-cpu-x64.zip")
dl(
    "https://github.com/ggml-org/llama.cpp/releases/download/b11429/llama-b11429-bin-win-cpu-x64.zip",
    z,
    "llama.cpp",
)
out = os.path.join(BASE, "tools", "llama")
os.makedirs(out, exist_ok=True)
with zipfile.ZipFile(z) as zf:
    zf.extractall(out)
log("[unzip] tools/llama")
for root, dirs, files in os.walk(out):
    for fn in sorted(files):
        if fn.endswith(".exe"):
            log("    " + os.path.relpath(os.path.join(root, fn), out))

log("=== 2/3 Android APK ===")
apk = os.path.join(BASE, "tools", "llama-node.apk")
dl(
    "https://github.com/adil-adysh/llama-cpp-android-node/releases/download/v0.2.0-b11429.r14/llama-cpp-android-node-v0.2.0-b11429.r14.apk",
    apk,
    "APK",
)

log("=== 3/3 检查模型 ===")
m = os.path.join(BASE, "models", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
if os.path.exists(m):
    log(f"模型存在 {os.path.getsize(m)/1024/1024:.0f}MB")
else:
    log("模型还没下完")
log("=== DONE ===")
