import os
import re
import subprocess
import time
import urllib.parse

from voders.sidequests import SideQuest, _register_side_quest


_YT_SEARCH_PREFIXES = {
    'youtube': 'ytsearch',
    'reddit': 'redditsearch',
    'bilibili': 'bilisearch',
}

_YT_SEARCH_URLS = {
    'tiktok': 'https://www.tiktok.com/search?q={q}',
    'snapchat': 'https://www.snapchat.com/spotlight/trending',
    'instagram': 'https://www.instagram.com/explore/tags/{tag}/',
    'facebook': 'https://www.facebook.com/watch/search/?q={q}',
    'twitter': 'https://x.com/search?q={q}&f=live',
    'x': 'https://x.com/search?q={q}&f=live',
}

_GALLERY_SEARCH_URLS = {
    'instagram': 'https://www.instagram.com/explore/tags/{tag}/',
    'pixiv': 'https://www.pixiv.net/en/tags/{tag}/artworks',
    'danbooru': 'https://danbooru.donmai.us/posts?tags={tag}',
    'gelbooru': 'https://gelbooru.com/index.php?page=post&s=list&tags={tag}',
    'yandere': 'https://yande.re/post?tags={tag}',
    'konachan': 'https://konachan.com/post?tags={tag}',
    'reddit': 'https://www.reddit.com/r/all/search?q={q}&restrict_sr=1&sort=relevance',
    'twitter': 'https://x.com/search?q={q}&f=live',
    'x': 'https://x.com/search?q={q}&f=live',
    'flickr': 'https://www.flickr.com/search/?text={q}',
    'pinterest': 'https://www.pinterest.com/search/pins/?q={q}',
    'artstation': 'https://www.artstation.com/search?sort_by=relevance&q={q}',
    'deviantart': 'https://www.deviantart.com/search?q={q}',
    'tumblr': 'https://www.tumblr.com/search/{q}',
    'wallhaven': 'https://wallhaven.cc/search?q={q}',
    'unsplash': 'https://unsplash.com/s/photos/{q}',
    'behance': 'https://www.behance.net/search/projects?search={q}',
    '500px': 'https://500px.com/search?q={q}',
    'imgur': 'https://imgur.com/search?q={q}',
    'vk': 'https://vk.com/search?c%5Bq%5D={q}&c%5Bsection%5D=auto',
    'weibo': 'https://s.weibo.com/weibo?q={q}',
}

_BROWSERS = ['chrome', 'brave', 'edge']

_DEFAULT_COUNT = 20
_MAX_COUNT = 100
_MIN_COUNT = 1


def _normalize_token(s):
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', s).strip('_').lower() or 'query'


def _quote_query(q):
    return urllib.parse.quote_plus(q.strip())


def _build_yt_url(platform, query, count):
    if platform in _YT_SEARCH_PREFIXES:
        return f"{_YT_SEARCH_PREFIXES[platform]}{count}:{query}"
    if platform in _YT_SEARCH_URLS:
        q = _quote_query(query)
        tag = _normalize_token(query)
        return _YT_SEARCH_URLS[platform].format(q=q, tag=tag)
    return f"https://{platform}.com/search?q={_quote_query(query)}"


def _build_gallery_url(platform, query):
    if platform in _GALLERY_SEARCH_URLS:
        q = _quote_query(query)
        tag = _normalize_token(query)
        return _GALLERY_SEARCH_URLS[platform].format(q=q, tag=tag)
    return f"https://{platform}.com/search?q={_quote_query(query)}"


def _format_duration(seconds):
    try:
        seconds = float(seconds)
        if seconds <= 0:
            return "N/A"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
    except Exception:
        return str(seconds) if seconds else "N/A"


def _run_yt_search(search_target, count, timeout=90):
    cmd = [
        'yt-dlp', search_target,
        '--flat-playlist',
        '--playlist-end', str(count),
        '--print', '%(title)s\t%(url)s\t%(extractor)s\t%(duration)s',
        '--no-warnings',
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return [], "yt-dlp is not installed. Install with: pip install yt-dlp"
    except subprocess.TimeoutExpired:
        return [], f"yt-dlp search timed out ({timeout}s)."
    if r.returncode != 0:
        err = (r.stderr or '').strip().splitlines()
        login_hints = ('Sign in', 'login', 'Login', 'restricted', 'age', 'private', 'cookies', 'Unable to extract')
        for browser in _BROWSERS:
            retry_cmd = list(cmd)
            retry_cmd.insert(1, '--cookies-from-browser')
            retry_cmd.insert(2, browser)
            try:
                rr = subprocess.run(retry_cmd, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                continue
            if rr.returncode == 0 and rr.stdout.strip():
                results = _parse_yt_output(rr.stdout)
                if results:
                    return results, None
        if any(any(h in line for h in login_hints) for line in err):
            return [], "Search failed — login-walled or restricted content. Cookies retry exhausted (Chrome/Brave/Edge)."
        last = err[-1] if err else "yt-dlp returned no results"
        return [], f"yt-dlp search failed: {last}"
    return _parse_yt_output(r.stdout), None


def _parse_yt_output(stdout):
    results = []
    if not stdout or not stdout.strip():
        return results
    for line in stdout.strip().split('\n'):
        fields = line.split('\t')
        while len(fields) < 4:
            fields.append('')
        title, url, extractor, duration = fields[0], fields[1], fields[2], fields[3]
        if not url:
            continue
        results.append({
            'title': title or '(no title)',
            'url': url,
            'platform': extractor or 'unknown',
            'duration_raw': duration,
            'duration': _format_duration(duration),
            'type': 'video/audio',
        })
    return results


def _run_gallery_search(search_url, count, timeout=120):
    base_cmd = ['gallery-dl', '-j', '--simulate']
    browsers = [None] + _BROWSERS
    last_err = None
    for browser in browsers:
        cmd = list(base_cmd)
        if browser is not None:
            cmd.extend(['--cookies-from-browser', browser])
        cmd.append(search_url)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            return [], "gallery-dl is not installed. Install with: pip install gallery-dl"
        except subprocess.TimeoutExpired:
            last_err = f"gallery-dl search timed out ({timeout}s)"
            continue
        if r.returncode == 0 and r.stdout.strip():
            results = _parse_gallery_output(r.stdout, count)
            if results:
                return results, None
            last_err = "gallery-dl returned no image entries"
        else:
            err = (r.stderr or '').strip()
            if 'No extractor found' in err or 'Unsupported URL' in err:
                return [], f"gallery-dl cannot search this URL: {err.splitlines()[-1] if err else 'unsupported'}"
            last_err = err.splitlines()[-1] if err else "gallery-dl returned no output"
            if browser is not None and browser != _BROWSERS[-1]:
                continue
    return [], last_err or "gallery-dl search produced no results"


def _parse_gallery_output(stdout, count):
    results = []
    seen_urls = set()
    if not stdout or not stdout.strip():
        return results
    for line in stdout.strip().split('\n'):
        try:
            import json
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        url = obj.get('url') or obj.get('_http_url') or ''
        if not url or url in seen_urls:
            continue
        title = (obj.get('title') or obj.get('name') or obj.get('id') or
                 obj.get('description') or obj.get('caption') or '')
        if isinstance(title, str):
            title = title.strip()[:200]
        else:
            title = str(title)[:200]
        category = obj.get('category') or obj.get('extractor') or 'gallery-dl'
        ext = (obj.get('extension') or obj.get('ext') or '').lower().lstrip('.')
        dimensions = ''
        width = obj.get('width')
        height = obj.get('height')
        if width and height:
            dimensions = f"{width}x{height}"
        seen_urls.add(url)
        results.append({
            'title': title or '(no title)',
            'url': url,
            'platform': category,
            'duration': dimensions or 'N/A',
            'duration_raw': f"{ext} {dimensions}".strip() if ext or dimensions else '',
            'type': f"image{f' ({ext})' if ext else ''}",
        })
        if len(results) >= count:
            break
    return results


def _safe_filename(s, maxlen=60):
    s = re.sub(r'[^A-Za-z0-9_\-]+', '_', s).strip('_')
    return (s or 'search')[:maxlen]


class Quest(SideQuest):
    name = 'media-search'
    description = 'Search media across platforms via yt-dlp (default, video/audio) or gallery-dl (with image keyword, images). Multi-platform via slash-separated list. Writes a results list file to results/downloads/others/.'

    def parse(self, args):
        use_image_engine = False
        platforms_raw = None
        query = None
        count = _DEFAULT_COUNT
        i = 0
        while i < len(args):
            a = args[i]
            al = a.lower()
            if al == 'image' and platforms_raw is None and query is None:
                use_image_engine = True
                i += 1
                continue
            if platforms_raw is None:
                if '/' in a:
                    parts = [p.strip().lower() for p in a.split('/') if p.strip()]
                    if not parts:
                        return None, "platform list is empty after splitting on '/'"
                    if any(not re.match(r'^[A-Za-z0-9_\-]+$', p) for p in parts):
                        return None, f"invalid platform name in '{a}' — only letters, digits, hyphen, underscore allowed"
                    platforms_raw = parts
                else:
                    if not re.match(r'^[A-Za-z0-9_\-]+$', al):
                        return None, f"invalid platform name '{a}' — only letters, digits, hyphen, underscore allowed"
                    platforms_raw = [al]
                i += 1
                continue
            if query is None:
                query = a.strip()
                i += 1
                continue
            try:
                count = int(al)
            except ValueError:
                return None, f"invalid count '{a}' — must be an integer between {_MIN_COUNT} and {_MAX_COUNT}"
            i += 1
        if not platforms_raw:
            return None, "quest media-search requires at least one platform name"
        if not query:
            return None, "quest media-search requires a search query"
        if count < _MIN_COUNT or count > _MAX_COUNT:
            return None, f"count must be between {_MIN_COUNT} and {_MAX_COUNT} (got {count})"
        seen = set()
        platforms = []
        for p in platforms_raw:
            if p not in seen:
                seen.add(p)
                platforms.append(p)
        return {
            'use_image_engine': use_image_engine,
            'platforms': platforms,
            'query': query,
            'count': count,
        }, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        use_image_engine = parsed['use_image_engine']
        platforms = parsed['platforms']
        query = parsed['query']
        count = parsed['count']
        engine_label = 'gallery-dl' if use_image_engine else 'yt-dlp'

        others_dir = os.path.join(results_dir, 'downloads', 'others')
        os.makedirs(others_dir, exist_ok=True)

        all_results = []
        per_platform_summary = []
        for platform in platforms:
            print(f"[media-search] {engine_label} → {platform}  query=\"{query}\"  cap={count}")
            if use_image_engine:
                search_url = _build_gallery_url(platform, query)
            else:
                search_url = _build_yt_url(platform, query, count)
            if use_image_engine:
                results, err = _run_gallery_search(search_url, count)
            else:
                results, err = _run_yt_search(search_url, count)
            if err:
                print(f"  → {platform}: {err}")
                per_platform_summary.append((platform, 0, err))
                continue
            for r in results:
                r['platform_source'] = platform
            all_results.extend(results)
            per_platform_summary.append((platform, len(results), None))
            print(f"  → {platform}: {len(results)} result(s)")

        if not all_results:
            print(f"\n[media-search] No results across any platform. No list file created.")
            print("\nPer-platform summary:")
            for p, n, e in per_platform_summary:
                status = f"{n} result(s)" if not e else f"failed — {e}"
                print(f"  {p}: {status}")
            return False

        list_path = self._write_results_file(
            others_dir, timestamp, engine_label, platforms, query, count,
            all_results, per_platform_summary,
        )
        print(f"\n[media-search] {len(all_results)} result(s) across {len(platforms)} platform(s).")
        print(f"[media-search] List file: {list_path}")
        print("\nPer-platform summary:")
        for p, n, e in per_platform_summary:
            status = f"{n} result(s)" if not e else f"failed — {e}"
            print(f"  {p}: {status}")
        if result_path:
            try:
                import shutil
                shutil.copy2(list_path, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True

    def _write_results_file(self, others_dir, timestamp, engine_label, platforms,
                            query, count, results, per_platform_summary):
        platform_str = '_'.join(platforms)
        safe_query = _safe_filename(query, 40)
        filename = f"voder_quest_media-search_{engine_label.replace('-', '')}_{platform_str}_{safe_query}_{timestamp}.txt"
        list_path = os.path.join(others_dir, filename)
        lines = []
        lines.append("=== VODER quest media-search RESULTS ===")
        lines.append(f"Engine: {engine_label}")
        lines.append(f"Platforms: {', '.join(platforms)}")
        lines.append(f"Query: {query}")
        lines.append(f"Per-platform cap: {count}")
        lines.append(f"Total results: {len(results)}")
        lines.append(f"Generated: {time.strftime('%Y/%m/%d %H:%M:%S')}")
        lines.append("")
        lines.append("Per-platform summary:")
        for p, n, e in per_platform_summary:
            status = f"{n} result(s)" if not e else f"failed — {e}"
            lines.append(f"  {p}: {status}")
        lines.append("")
        lines.append("Use 'quest download \"<url>\"' to fetch as audio (default) or video.")
        lines.append("Use 'quest download image \"<url>\"' to fetch as image.")
        lines.append("")
        for i, r in enumerate(results, 1):
            lines.append(f"--- Entry {i} of {len(results)} ---")
            lines.append(f"Source platform: {r['platform_source']}")
            lines.append(f"Title: {r['title']}")
            lines.append(f"URL: {r['url']}")
            lines.append(f"Extractor/Platform: {r['platform']}")
            lines.append(f"Type: {r['type']}")
            lines.append(f"Duration/Dimensions: {r['duration']}")
            if r.get('duration_raw'):
                lines.append(f"Detail: {r['duration_raw']}")
            lines.append("")
        with open(list_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return list_path


_register_side_quest(Quest)
