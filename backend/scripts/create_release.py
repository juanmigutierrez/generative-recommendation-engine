"""
Create the `v1.0-artifacts` GitHub release and upload everything in release/ -- without the
GitHub CLI. Needs a personal access token with the `repo` scope (classic) or
"Contents: read and write" (fine-grained): https://github.com/settings/tokens

    python backend/scripts/make_release_bundle.py          # fills release/
    set GITHUB_TOKEN=ghp_...                               # PowerShell: $env:GITHUB_TOKEN="ghp_..."
    python backend/scripts/create_release.py

Re-running is safe: an existing release is reused and already-uploaded files are skipped.
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

OWNER, REPO, TAG = "juanmigutierrez", "generative-recommendation-engine", "v1.0-artifacts"
TITLE = "Pre-computed artifacts for the tutorials"
API = "https://api.github.com"


def call(url, data=None, method=None, content_type="application/json", raw=False):
    tok = os.environ.get("GITHUB_TOKEN")
    if not tok:
        sys.exit("set GITHUB_TOKEN first (https://github.com/settings/tokens, scope: repo)")
    body = data if raw else (json.dumps(data).encode() if data is not None else None)
    req = urllib.request.Request(url, data=body, method=method or ("POST" if body else "GET"))
    req.add_header("Authorization", f"Bearer {tok}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Content-Type", content_type)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read() or b"{}")


def main():
    files = sorted(os.listdir("release")) if os.path.isdir("release") else []
    if not files:
        sys.exit("release/ is empty -- run backend/scripts/make_release_bundle.py first")
    try:
        rel = call(f"{API}/repos/{OWNER}/{REPO}/releases/tags/{TAG}")
        print(f"release {TAG} already exists -> adding missing files")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        rel = call(f"{API}/repos/{OWNER}/{REPO}/releases",
                   {"tag_name": TAG, "name": TITLE, "target_commitish": "main",
                    "body": "Processed data, embeddings and trained checkpoints used by notebooks/tutorial_0*.ipynb"})
        print(f"created release {TAG}")
    have = {a["name"] for a in rel.get("assets", [])}
    upload_base = rel["upload_url"].split("{")[0]
    for f in files:
        if f in have:
            print(f"  skip (already uploaded) {f}"); continue
        p = os.path.join("release", f)
        with open(p, "rb") as fh:
            data = fh.read()
        print(f"  uploading {f} ({len(data)/1e6:.1f} MB) ...", end=" ", flush=True)
        call(f"{upload_base}?name={urllib.parse.quote(f)}", data, content_type="application/octet-stream", raw=True)
        print("ok")
    print(f"\ndone: https://github.com/{OWNER}/{REPO}/releases/tag/{TAG}")


if __name__ == "__main__":
    main()
