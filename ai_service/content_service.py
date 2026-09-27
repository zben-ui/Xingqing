from html.parser import HTMLParser
from pathlib import Path


class _ArticleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.capture = ""
        self.parts = {"h2": [], "p": []}
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        if tag == "div" and "article-content" in dict(attrs).get("class", "").split():
            self.inside = True
        if self.inside and tag in self.parts:
            self.capture = tag
            if tag == "p":
                self.parts["p"] = []

    def handle_endtag(self, tag):
        if tag == "p" and self.capture == "p":
            self.paragraphs.append("".join(self.parts["p"]).strip())
        if tag == self.capture:
            self.capture = ""
        if tag == "div":
            self.inside = False

    def handle_data(self, data):
        if self.capture:
            self.parts[self.capture].append(data)


class ContentService:
    def __init__(self, directory: Path):
        self.directory = directory

    def articles(self):
        result = []
        for index in range(1, 5):
            try:
                parser = _ArticleParser()
                parser.feed((self.directory / f"article{index}.html").read_text(encoding="utf-8-sig"))
            except OSError:
                continue
            # Generic template filler follows the literary main paragraph in the supplied HTML.
            content = parser.paragraphs[0] if parser.paragraphs else ""
            if not content:
                continue
            result.append({
                "id": f"local-{index}", "title": "".join(parser.parts["h2"]).strip(),
                "summary": content[:80] + "…", "content": content,
                "cover_url": f"/static/assets/articles/{index}.png",
                "kind": "情绪陪伴散文", "created_at": "",
                "disclaimer": "文学阅读内容，不是心理治疗或医疗建议。",
            })
        return result
