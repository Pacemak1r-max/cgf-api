"""
Step 1: pull videos for a niche from Oriane and flatten them.

Usage:
    export ORIANE_KEY="你的key"
    python oriane_fetch.py "skincare" --platforms tiktok instagram --days 60 --limit 100

Outputs:
    data/<niche>_raw.json      原始响应（备份）
    data/<niche>_videos.json   扁平化后的视频列表（给下一步用）
"""
import argparse, json, os, sys, datetime, re
import requests

BASE = "https://connect.oriane.xyz"
KEY = os.environ.get("ORIANE_KEY")
if not KEY:
    sys.exit("请先 export ORIANE_KEY=你的key")

# 文档里没写认证 header 名，两种都试
AUTH_HEADERS = [
    {"Authorization": f"Bearer {KEY}"},
    {"x-api-key": KEY},
]


def leaf(field, kw, platforms, after, min_views, lang):
    """一个叶子查询：某个字段模糊包含关键词 + 公共过滤条件（叶子内部是 AND）"""
    filters = {
        field: {"includesExactly": {"values": [kw], "operator": "or"}},
        "platform": {"includes": platforms},
        "format": {"includes": ["video"]},
        "publishedAt": {"after": after},
        "viewsCount": {"min": min_views},
    }
    if lang:
        filters["transcriptLanguage"] = {"includes": [lang]}
    return {"name": field, "operator": "and", "filters": filters}


def search(kw, platforms, days, limit, min_views, lang, projection="full"):
    after = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    body = {
        "operator": "or",   # transcript / caption / hashtags 任一命中即可
        "queries": [
            leaf("transcript", kw, platforms, after, min_views, lang),
            leaf("caption", kw, platforms, after, min_views, lang),
            leaf("hashtags", kw, platforms, after, min_views, lang),
        ],
    }
    params = {"sort": "viewsCount:desc", "limit": limit, "offset": 0, "projection": projection}
    last_err = None
    for h in AUTH_HEADERS:
        r = requests.post(f"{BASE}/rest/contents/search", params=params, json=body,
                          headers={**h, "Content-Type": "application/json"}, timeout=60)
        if r.status_code in (200, 206):
            print(f"[ok] auth header = {list(h)[0]}, status {r.status_code}")
            return r.json()
        last_err = (r.status_code, r.text[:500])
        print(f"[retry] header {list(h)[0]} -> {r.status_code}: {r.text[:200]}")
    sys.exit(f"请求失败: {last_err}")


def relevant(v, kw):
    words = re.findall(r"[a-z0-9]+", kw.lower())
    text = " ".join([v.get("transcript") or "", v.get("caption") or "",
                     " ".join(v.get("on_screen_text") or [])]).lower()
    tags = [h.lower().lstrip("#") for h in (v.get("hashtags") or [])]
    for w in words:
        in_text = re.search(rf"\b{re.escape(w)}(s|es)?\b", text) is not None
        in_tag = any(t.startswith(w) for t in tags)
        if not (in_text or in_tag):
            return False
    return True


def flatten(res):
    """把一条 full projection 结果压成后续分析需要的字段"""
    media = res.get("media") or []
    transcript = " ".join((m.get("transcript") or "") for m in media).strip()
    ocr = []
    for m in media:
        ocr += (m.get("onScreenText") or [])
    frames = []
    for m in media:
        for f in (m.get("frames") or [])[:3]:
            if f.get("url"):
                frames.append(f["url"])
    comments = [
        {"text": c.get("content"), "likes": c.get("likesCount")}
        for c in (res.get("popularComments") or [])[:8]
    ]
    duration = sum((m.get("duration") or 0) for m in media)
    views = res.get("viewsCount") or 0
    likes = res.get("likesCount") or 0
    er = res.get("engagementRatePerViews")
    if er is None and views:
        er = 100.0 * (res.get("interactionsCount") or likes) / views

    url = None
    if res.get("platform") == "tiktok" and res.get("profileHandle") and res.get("platformId"):
        url = f"https://www.tiktok.com/@{res['profileHandle']}/video/{res['platformId']}"
    m = re.search(r"/instagram/([A-Za-z0-9_-]+)/", res.get("thumbnailMediaUrl") or "")
    if res.get("platform") == "instagram" and m:
        url = f"https://www.instagram.com/reel/{m.group(1)}/"

    return {
        "id": res.get("id"),
        "platform": res.get("platform"),
        "handle": res.get("profileHandle"),
        "followers": res.get("profileFollowersCount"),
        "publishedAt": res.get("publishedAt"),
        "views": views,
        "likes": likes,
        "comments": res.get("commentsCount"),
        "shares": res.get("sharesCount"),
        "er_views": er,                       # 互动率（按播放），百分比
        "er_followers": res.get("engagementRatePerFollowers"),
        "duration": duration,
        "caption": (res.get("caption") or "")[:600],
        "hashtags": res.get("hashtags") or [],
        "audio_title": res.get("audioTitle"),
        "audio_author": res.get("audioAuthor"),
        "transcript": transcript[:2500],
        "on_screen_text": ocr[:40],
        "popular_comments": comments,
        "thumbnail": res.get("thumbnailMediaUrl"),
        "frames": frames[:3],
        "url": url,
        "matchedQueries": res.get("matchedQueries"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("niche")
    ap.add_argument("--platforms", nargs="+", default=["tiktok", "instagram"])
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--min-views", type=int, default=1000)
    ap.add_argument("--lang", default="en", help="transcript 语言，传空字符串 '' 则不过滤")
    a = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", a.niche.lower()).strip("_")

    raw = search(a.niche, a.platforms, a.days, a.limit, a.min_views, a.lang)
    json.dump(raw, open(f"data/{slug}_raw.json", "w"), ensure_ascii=False, indent=1)

    results = (raw.get("data") or {}).get("results") or []
    videos = [flatten(r) for r in results]
    # 去掉没有 transcript 也没有屏幕字的（没法打标签）
    videos = [v for v in videos if v["transcript"] or v["on_screen_text"]]
    json.dump(videos, open(f"data/{slug}_videos.json", "w"), ensure_ascii=False, indent=1)

    print(f"拉到 {len(results)} 条，可用 {len(videos)} 条 -> data/{slug}_videos.json")
    print("pagination:", (raw.get("metadata") or {}).get("pagination"))
    if videos:
        print("示例:", json.dumps(videos[0], ensure_ascii=False)[:800])
