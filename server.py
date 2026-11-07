"""
实时查询服务：GET /analyze?niche=xxx  -> 拉数据 -> 并行打标签 -> 找空白 -> 返回 results json
查过的领域缓存在 data/ 下，第二次秒出。

启动（和两个脚本放同一目录，环境变量和之前一样）：
    pip3 install flask
    python3 server.py
然后另开终端做隧道（任选）：
    brew install cloudflared && cloudflared tunnel --url http://localhost:8000
拿到 https://xxxx.trycloudflare.com 这个地址给 Lovable 用。
"""
import os, re, json, threading, time
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify

import oriane_fetch as of
import gap_pipeline as gp

app = Flask(__name__)
LOCK = threading.Lock()
RUNNING = {}          # slug -> 状态
os.makedirs("data", exist_ok=True)


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def analyze(niche, limit=40, workers=8):
    slug = slugify(niche)
    out_path = f"data/{slug}_results.json"
    if os.path.exists(out_path):
        return json.load(open(out_path))

    t0 = time.time()
    raw = of.search(niche, ["tiktok", "instagram"], 60, 100, 1000, "en")
    results = (raw.get("data") or {}).get("results") or []
    videos = [of.flatten(r) for r in results]
    videos = [v for v in videos if v["transcript"] or v["on_screen_text"]]
    videos = [v for v in videos if of.relevant(v, niche)]
    videos = sorted(videos, key=lambda v: -(v["views"] or 0))[:limit]
    print(f"[{niche}] fetched {len(results)}, relevant {len(videos)}")
    if len(videos) < 8:
        return {"error": f"该领域只找到 {len(videos)} 条可用视频，换个更常见的关键词试试", "niche": niche}

    # 并行打标签
    with ThreadPoolExecutor(max_workers=workers) as ex:
        labels = list(ex.map(gp.label_video, videos))
    for v, l in zip(videos, labels):
        v["format"] = l
    json.dump(videos, open(f"data/{slug}_videos.json", "w"), ensure_ascii=False, indent=1)

    rows, base_er = gp.aggregate(videos)
    gaps, early, proven, saturated = gp.find_gaps(rows)
    with ThreadPoolExecutor(max_workers=4) as ex:
        cards = list(ex.map(lambda g: gp.recommend(niche, g, videos), gaps + early + proven))

    res = {
        "niche": niche,
        "sample_size": len(videos),
        "niche_median_er": round(base_er, 2),
        "formats": rows,
        "cards": cards,
        "saturated": saturated,
        "top_videos": [{"id": v["id"], "handle": v["handle"], "platform": v["platform"], "format": v["format"],
                        "views": v["views"], "er": round(v["er_views"] or 0, 2),
                        "thumbnail": v["thumbnail"], "url": v["url"]}
                       for v in sorted(videos, key=lambda v: -(v["er_views"] or 0))[:10]],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1)
    return res


@app.route("/analyze")
def route_analyze():
    niche = (request.args.get("niche") or "").strip()
    if not niche:
        return jsonify({"error": "niche required"}), 400
    slug = slugify(niche)
    with LOCK:
        if slug in RUNNING:
            return jsonify({"status": "running", "niche": niche}), 202
        RUNNING[slug] = True
    try:
        res = analyze(niche, limit=int(request.args.get("limit", 40)))
        return jsonify(res), (200 if "error" not in res else 422)
    except Exception as e:
        return jsonify({"error": str(e)[:300], "niche": niche}), 500
    finally:
        with LOCK:
            RUNNING.pop(slug, None)


@app.route("/niches")
def route_niches():
    """已缓存的领域列表，给下拉菜单用"""
    out = []
    for f in sorted(os.listdir("data")):
        if f.endswith("_results.json"):
            d = json.load(open(f"data/{f}"))
            out.append({"niche": d.get("niche"), "slug": f.replace("_results.json", "")})
    return jsonify(out)


@app.route("/health")
def health():
    return jsonify({"ok": True})


EXPLAIN_PROMPT = """You are a short-form video coach. A creator wants to make videos like this one.
Niche: {niche}. Format: {fmt}. Platform: {platform}. Views: {views}. Engagement rate: {er}%.
Transcript: {transcript}
On-screen text: {ocr}
Audio: {audio}
Caption: {caption}

Return ONLY a JSON object with these keys:
{{"hook": "what the first 3 seconds do and why it stops the scroll (1-2 sentences)",
  "structure": "beat-by-beat structure of the video (3-5 short beats)",
  "why_it_works": "2 sentences on why this earns engagement",
  "how_to_start": ["step 1", "step 2", "step 3"],
  "shot_list": ["shot 1", "shot 2", "shot 3", "shot 4"],
  "gear_and_edit": "1-2 sentences: what you need and the editing style",
  "your_twist": "one concrete way to make it your own instead of copying"}}"""


@app.route("/explain")
def route_explain():
    niche = (request.args.get("niche") or "").strip()
    vid = (request.args.get("id") or "").strip()
    if not niche or not vid:
        return jsonify({"error": "niche and id required"}), 400
    cache = f"data/explain_{re.sub(r'[^A-Za-z0-9_]', '_', vid)}.json"
    if os.path.exists(cache):
        return jsonify(json.load(open(cache)))
    path = f"data/{slugify(niche)}_videos.json"
    if not os.path.exists(path):
        return jsonify({"error": "niche not analyzed yet"}), 404
    v = next((x for x in json.load(open(path)) if x.get("id") == vid), None)
    if not v:
        return jsonify({"error": "video not found"}), 404
    prompt = EXPLAIN_PROMPT.format(
        niche=niche, fmt=gp.FORMAT_LABELS_EN.get(v.get("format"), v.get("format")),
        platform=v["platform"], views=v["views"], er=round(v["er_views"] or 0, 2),
        transcript=(v["transcript"] or "(none)")[:1500],
        ocr=" | ".join(v["on_screen_text"][:20]) or "(none)",
        audio=f"{v.get('audio_title') or ''} {v.get('audio_author') or ''}".strip() or "(none)",
        caption=(v["caption"] or "")[:300])
    try:
        out = gp.chat([{"role": "user", "content": prompt}], temperature=0.5, max_tokens=2500)
        m = re.search(r"\{.*\}", out, re.S)
        data = json.loads(m.group(0)) if m else {"error": "could not parse"}
    except Exception as e:
        return jsonify({"error": str(e)[:300]}), 500
    data.update({"id": vid, "handle": v["handle"], "url": v["url"], "thumbnail": v["thumbnail"]})
    json.dump(data, open(cache, "w"), ensure_ascii=False, indent=1)
    return jsonify(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), threaded=True)
