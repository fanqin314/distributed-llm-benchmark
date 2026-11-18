import urllib.request, json, os, zipfile, time, sys

BASE = r"D:/workbuddy/Decentralized AI"
def log(m):
    print(m, flush=True)

def download(url, dest, label):
    if os.path.exists(dest) and os.path.getsize(dest) > 1024*1024:
        log(f"[跳过] {label} 已存在 {os.path.getsize(dest)/1024/1024:.0f}MB")
        return True
    t0 = time.time()
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as r, open(dest, 'wb') as f:
        total = int(r.headers.get('Content-Length') or 0)
        got = 0
        last = 0
        while True:
            b = r.read(1024*1024)
            if not b: break
            f.write(b); got += len(b)
            if time.time()-last > 5:
                sp = got/(time.time()-t0)/1024/1024
                pct = f"{got/total*100:.1f}%" if total else "?"
                log(f"[{label}] {got/1024/1024:.0f}/{total/1024/1024:.0f}MB ({pct}) {sp:.2f}MB/s")
                last = time.time()
    log(f"[完成] {label} {os.path.getsize(dest)/1024/1024:.0f}MB 用时{time.time()-t0:.0f}s")
    return True

log("=== 1. 查询 llama.cpp 最新版 ===")
try:
    req = urllib.request.Request('https://api.github.com/repos/ggml-org/llama.cpp/releases/latest',
                                 headers={'User-Agent':'Mozilla/5.0','Accept':'application/vnd.github+json'})
    rel = json.loads(urllib.request.urlopen(req, timeout=20).read())
    tag = rel['tag_name']
    log(f"最新版本: {tag}")
    want = None
    for a in rel['assets']:
        n = a['name']
        if 'win-cpu-x64' in n and n.endswith('.zip'):
            want = (a['name'], a['browser_download_url'], a['size']); break
    if not want:
        log("!! 未找到 win-cpu-x64 资产，列出可用资产:")
        for a in rel['assets'][:25]: log("   " + a['name'])
    else:
        log(f"目标: {want[0]} ({want[2]/1024/1024:.0f}MB)")
        z = os.path.join(BASE, "tools", want[0])
        download(want[1], z, "llama.cpp")
        outdir = os.path.join(BASE, "tools", "llama")
        os.makedirs(outdir, exist_ok=True)
        with zipfile.ZipFile(z) as zf: zf.extractall(outdir)
        log(f"[解压] → {outdir}")
        for root, dirs, files in os.walk(outdir):
            for fn in files:
                if fn.endswith('.exe'):
                    log("   exe: " + os.path.relpath(os.path.join(root,fn), outdir))
except Exception as e:
    log(f"!! llama.cpp 阶段失败: {type(e).__name__}: {e}")

log("=== 2. 下载 Qwen2.5-1.5B-Instruct Q4 ===")
try:
    u = ('https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/repo'
         '?Revision=master&FilePath=qwen2.5-1.5b-instruct-q4_k_m.gguf')
    dest = os.path.join(BASE, "models", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    download(u, dest, "1.5B模型")
except Exception as e:
    log(f"!! 模型阶段失败: {type(e).__name__}: {e}")
log("=== ALL DONE ===")
