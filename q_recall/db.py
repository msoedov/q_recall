from pathlib import Path


class ParadigmDB:
    def __init__(self):
        self.sources = {}  # name -> dir path

    def register_fs(self, name: str, path: str):
        """Register a directory as a source."""
        self.sources[name] = Path(path)
        if not self.sources[name].exists():
            raise ValueError(f"Source {name} does not exist: {path}")

    def resolve(self, pattern: str, source: str = None):
        bases = [self.sources[source]] if source else self.sources.values()
        for base in bases:
            yield from base.glob(pattern)

    def read(self, uri: str) -> str:
        p = Path(uri.replace("file://", ""))
        return p.read_text(encoding="utf-8", errors="ignore")
