"""Bound retained command output while draining the entire pipe."""

BYTE_LIMIT = 50 * 1024
LINE_LIMIT = 2000


class OutputTail:
    def __init__(self):
        self.data = bytearray()
        self.total = 0
        self.truncated = False

    def feed(self, chunk):
        self.total += len(chunk)
        self.data.extend(chunk)
        # Leave room for the truncation notice in the returned output.
        capacity = BYTE_LIMIT - 128
        if len(self.data) > capacity:
            del self.data[:-capacity]
            self.truncated = True
        lines = self.data.count(b"\n") + bool(self.data and self.data[-1] != 10)
        limit = LINE_LIMIT - 1 if self.truncated else LINE_LIMIT
        while lines > limit:
            del self.data[: self.data.index(10) + 1]
            lines -= 1
            self.truncated = True
            limit = LINE_LIMIT - 1

    def text(self):
        # Ignore an incomplete leading UTF-8 character cut by the byte boundary.
        text = bytes(self.data).decode("utf-8", errors="ignore")
        if self.truncated:
            return f"[output truncated; {self.total} bytes produced]\n" + text
        return text
