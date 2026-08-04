"""Post structure and publishing targets.

    from quantpost.render import Post, Section, publish
    post = Post(title="...", slug="...").add("Section", "body", figures=[fig])
    assert not post.audit()
    publish.hugo_page_bundle(post)
    publish.medium_bundle(post)
"""

from . import publish
from .post import Post, Section

__all__ = ["Post", "Section", "publish"]
