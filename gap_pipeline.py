"""
Step 2: 打形式标签 -> 聚合 -> 找内容空白 -> 生成选题 -> 输出给 Lovable 的 results.json

Usage:
    export LLM_BASE_URL="https://api.openai.com/v1"      # 或 GMI Cloud 的 OpenAI 兼容地址
    export LLM_API_KEY="..."
    export LLM_MODEL="gpt-4o-mini"                        # 或 GMI 上的模型名
    python gap_pipeline.py data/skincare_videos.json --niche "skincare"

Output:
    data/skincare_results.json   直接贴进 Lovable
"""
import argparse, json, os, sys, statistics, re
import requests

BASE = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
KEY = os.environ.get("LLM_API_KEY")
MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
if not KEY:
    sys.exit("请先 export LLM_API_KEY / LLM_BASE_URL / LLM_MODEL")

# 固定标签表：不要让 LLM 自由发挥，否则聚合不出来
FORMATS = [
    "talking_head_review",   # 口播测评/观点
    "step_by_step_tutorial", # 分步教程
    "before_after",          # 前后对比
    "skit_story",            # 剧情/短片
    "day_in_life_vlog",      # 日常 vlog
    "unboxing_haul",         # 开箱/haul
    "street_interview",      # 街头采访
    "data_explainer",        # 数据/科普讲解
    "contrarian_take",       # 反常识观点/辟谣
    "pov_trend_audio",       # 跟音频/POV 梗
]

FORMAT_LABELS_ZH = {
    "talking_head_review": "口播测评", "step_by_step_tutorial": "分步教程",
    "before_after": "前后对比", "skit_story": "剧情短片", "day_in_life_vlog": "日常vlog",
    "unboxing_haul": "开箱/haul", "street_interview": "街头采访", "data_explainer": "数据科普",
    "contrarian_take": "反常识观点", "pov_trend_audio": "跟音频/POV梗",
}
FORMAT_LABELS_EN = {
    "talking_head_review": "Talking-head review", "step_by_step_tutorial": "Step-by-step tutorial",
    "before_after": "Before & after", "skit_story": "Skit / story", "day_in_life_vlog": "Day-in-life vlog",
    "unboxing_haul": "Unboxing / haul", "street_interview": "Street interview", "data_explainer": "Data explainer",
    "contrarian_take": "Contrarian take", "pov_trend_audio": "POV / trend audio",
}


def chat(messages, temperature=0, max_tokens=400):
    r = requests.post(f"{BASE}/chat/completions",
                      headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
                      json={"model": MODEL, "messages": messages,
                            "temperature": temperature, "max_tokens": max_tokens},
                      timeout=90)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    # 思考型模型可能把正文留空、只返回 reasoning_content，兜底用它
    content = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
    return content.strip()


def label_video(v):
    prompt = f"""You classify short-form social videos into ONE format label.

Allowed labels (pick exactly one): {", ".join(FORMATS)}

Video signals:
- transcript (what is said): {v['transcript'][:1200] or '(none)'}
- on-screen text: {" | ".join(v['on_screen_text'][:15]) or '(none)'}
- caption: {v['caption'][:300] or '(none)'}
- audio: {v.get('audio_title') or ''} {v.get('audio_author') or ''}
- duration: {v.get('duration')}s

Answer with the label only."""
    try:
        out = chat([{"role": "user", "content": prompt}], max_tokens=1500).lower()
    except Exception as e:
        print("   [warn] LLM 调用失败:", str(e)[:120])
        out = ""
    # 取最后出现的标签（思考型模型会在推理里提到多个，结论在最后）
    best, pos = None, -1
    for f in FORMATS:
        p = out.rfind(f)
        if p > pos:
            best, pos = f, p
    return best or "talking_head_review"  # 兜底


def aggregate(videos):
    base_er = statistics.median(v["er_views"] for v in videos if v["er_views"] is not None)
    groups = {}
    for v in videos:
        groups.setdefault(v["format"], []).append(v)
    n_total = len(videos)
    rows = []
    for f, vs in groups.items():
        ers = [x["er_views"] for x in vs if x["er_views"] is not None]
        med = statistics.median(ers) if ers else 0
        rows.append({
            "format": f,
            "label": FORMAT_LABELS_EN.get(f, f),
            "label_zh": FORMAT_LABELS_ZH.get(f, f),
            "count": len(vs),
            "share": round(len(vs) / n_total, 3),          # x 轴：占比
            "median_er": round(med, 2),                    # y 轴：中位互动率 %
            "perf": round(med / base_er, 2) if base_er else 0,  # 相对领域中位数
            "median_views": int(statistics.median(x["views"] for x in vs)),
            "distinct_creators": len({x["handle"] for x in vs}),
        })
    rows.sort(key=lambda r: -r["count"])
    return rows, base_er


def find_gaps(rows, min_count=3, min_perf=1.2, max_share=0.15):
    for r in rows:
        r["gap_score"] = round(r["perf"] / max(r["share"], 0.01), 1)
    # 空白：少见（≤15%）、表现好（≥1.2x）、至少 3 条且 ≥2 个账号
    gaps = [r for r in rows
            if r["count"] >= min_count and r["perf"] >= min_perf and r["share"] <= max_share
            and r["distinct_creators"] >= 2]
    # 早期信号：只有 2 条但表现极好（≥3x）——样本太小，标注为"待验证"
    early = [r for r in rows
             if r["count"] == 2 and r["perf"] >= 3 and r["distinct_creators"] >= 2]
    # 已验证仍有空间：占比不算低，但表现远高于平均（≥2x）
    proven = [r for r in rows
              if r["share"] > max_share and r["perf"] >= 2 and r not in gaps]
    # 饱和：又多又差
    saturated = [r for r in rows if r["share"] >= 0.2 and r["perf"] <= 0.9]
    for lst in (gaps, early, proven):
        lst.sort(key=lambda r: -r["gap_score"])
    for r, tier in [(x, "gap") for x in gaps] + [(x, "early") for x in early] + [(x, "proven") for x in proven]:
        r["tier"] = tier
    # 兜底：一个机会都没找到时，把表现最好的 2 种形式（≥2 条）当"最佳选择"给出
    if not (gaps or early or proven):
        proven = sorted([r for r in rows if r["count"] >= 2], key=lambda r: -r["perf"])[:2]
        for r in proven:
            r["tier"] = "proven"
    # 兜底：没有饱和形式时，把占比最高且表现不突出的那个标为拥挤
    if not saturated:
        top = max(rows, key=lambda r: r["share"])
        if top["share"] >= 0.3 and top["perf"] < 1.5:
            saturated = [top]
    return gaps[:3], early[:2], proven[:2], saturated


def pick_evidence(videos, fmt, k=3):
    """证据视频：同形式、互动率最高、transcript 够长（有内容可看）、去掉跨平台重复发布的同一条"""
    cands = sorted([v for v in videos if v["format"] == fmt], key=lambda v: -(v["er_views"] or 0))
    out, seen = [], set()
    for v in cands:
        text = v["transcript"] or " ".join(v["on_screen_text"])
        if len(text) < 40:
            continue
        key = text[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
        if len(out) == k:
            break
    return out or cands[:k]


def extract_json(text):
    """从模型输出里找最后一个含 "title" 的 JSON 对象"""
    for m in reversed(list(re.finditer(r"\{[^{}]*\"title\"[^{}]*\}", text, re.S))):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return None


def recommend(niche, gap, videos):
    evid = pick_evidence(videos, gap["format"])
    evid_txt = "\n".join(
        f"- @{v['handle']} ({v['views']} views, ER {v['er_views']:.1f}%): "
        f"{(v['transcript'] or ' '.join(v['on_screen_text']))[:300]}"
        for v in evid)
    prompt = f"""You are a short-form video strategist. Niche: {niche}.
The format "{gap['label']}" is only {gap['share']*100:.0f}% of the top videos in this niche,
but its median engagement rate is {gap['perf']}x the niche median.

Best-performing examples of this format:
{evid_txt}

Propose ONE concrete video a creator could shoot this week in this format, for this niche.
Respond with ONLY a JSON object, no markdown, exactly these keys:
{{"title": "catchy video title", "how_to_shoot": "2-3 sentences, max 50 words", "why_it_works": "1-2 sentences, max 30 words, cite the numbers above"}}"""
    idea = None
    for attempt in range(2):
        try:
            out = chat([{"role": "user", "content": prompt}], temperature=0.7, max_tokens=2500)
            idea = extract_json(out)
            if idea and idea.get("how_to_shoot"):
                break
        except Exception as e:
            print("   [warn] 选题生成失败:", str(e)[:120])
    if not idea or not idea.get("how_to_shoot"):
        idea = {"title": f"{gap['label']} for {niche}",
                "how_to_shoot": "Follow the structure of the top examples; keep it under 30s.",
                "why_it_works": f"{gap['perf']}x the niche's median engagement, only {gap['share']*100:.0f}% of top videos."}
    return {
        **gap,
        "idea": idea,
        "evidence": [{"id": v["id"], "handle": v["handle"], "platform": v["platform"], "views": v["views"],
                      "er": round(v["er_views"] or 0, 2), "thumbnail": v["thumbnail"],
                      "url": v["url"], "snippet": (v["transcript"] or " ".join(v["on_screen_text"]))[:160]}
                     for v in evid],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("videos_json")
    ap.add_argument("--niche", required=True)
    a = ap.parse_args()

    videos = json.load(open(a.videos_json))
    print(f"{len(videos)} 条视频，开始打标签（{MODEL}）...")
    for i, v in enumerate(videos):
        if not v.get("format"):
            v["format"] = label_video(v)
        print(f"  {i+1}/{len(videos)} {v['format']}  @{v['handle']}")
        if (i + 1) % 10 == 0:   # 每 10 条存一次，中断了可以接着跑
            json.dump(videos, open(a.videos_json, "w"), ensure_ascii=False, indent=1)
    json.dump(videos, open(a.videos_json, "w"), ensure_ascii=False, indent=1)  # 缓存标签

    rows, base_er = aggregate(videos)
    gaps, early, proven, saturated = find_gaps(rows)
    print("\n形式分布：")
    for r in rows:
        print(f"  {r['label_zh']:<8} n={r['count']:>3} share={r['share']:.0%} ER={r['median_er']}% perf={r['perf']}x")
    print("\n空白：", [g["label_zh"] for g in gaps],
          " 早期信号：", [g["label_zh"] for g in early],
          " 已验证仍有空间：", [g["label_zh"] for g in proven],
          " 饱和：", [s["label_zh"] for s in saturated])

    cards = [recommend(a.niche, g, videos) for g in gaps + early + proven]
    results = {
        "niche": a.niche,
        "sample_size": len(videos),
        "niche_median_er": round(base_er, 2),
        "formats": rows,            # 散点图数据：x=share, y=median_er, tier 标记
        "cards": cards,             # 选题卡片，每张带 tier: gap / early / proven
        "saturated": saturated,     # 饱和形式
        "top_videos": [{"id": v["id"], "handle": v["handle"], "platform": v["platform"], "format": v["format"],
                        "views": v["views"], "er": round(v["er_views"] or 0, 2),
                        "thumbnail": v["thumbnail"], "url": v["url"]}
                       for v in sorted(videos, key=lambda v: -(v["er_views"] or 0))[:10]],
    }
    out = a.videos_json.replace("_videos.json", "_results.json")
    json.dump(results, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\n写入 {out}，直接贴进 Lovable")
