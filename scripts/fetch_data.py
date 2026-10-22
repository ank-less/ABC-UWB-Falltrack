"""Download selected files of the official dataset zip (abc2027_uwb_train_v1.zip).

The zip is read with HTTP range requests, so only the requested members are
transferred: no 8 GB download, no unzip step. Each file is checked against the
CRC32 and size stored in the zip. The source is the Google Drive link published on
the challenge's "Dataset and Tutorial" page.

Subsets:
  all    every stream (13 train + 5 val), the CSVs, README/LICENSE and the package code
  smoke  the same files but only train_00, train_01, val_00, val_01
The segments/ folder (aligned source recordings) is skipped: the package README forbids
training or inference on the source LIARA recordings.
"""

from __future__ import annotations

import argparse
import io
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

FILE_ID = "1gJfL1YtObCs9JBFjPrC8qYq9a7L2kt2u"
URL = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t"
SMOKE_STREAMS = ("train/stream_00.h5", "train/stream_01.h5", "val/stream_00.h5", "val/stream_01.h5")


def _get(start: int, end: int, timeout: float = 120.0):
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}", "User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(request, timeout=timeout)


def remote_size() -> int:
    """Total size from ``Content-Range`` (Drive answers HEAD with length 0)."""
    with _get(0, 0, timeout=60) as response:
        content_range = response.headers.get("Content-Range", "")
    if "/" not in content_range:
        raise SystemExit(f"the download link did not answer a range request (Content-Range={content_range!r})")
    return int(content_range.rsplit("/", 1)[1])


class RangeFile(io.RawIOBase):
    """Seekable read-only view of the remote zip."""

    def __init__(self, size: int):
        self.size, self.pos = size, 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def readinto(self, buffer) -> int:
        if self.pos >= self.size:
            return 0
        with _get(self.pos, min(self.size, self.pos + len(buffer)) - 1) as response:
            data = response.read()
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


def wanted(name: str, subset: str) -> bool:
    if name.endswith("/") or "/segments/" in name:
        return False
    if "/streams/" in name:
        return subset == "all" or any(name.endswith(s) for s in SMOKE_STREAMS)
    return True


def extract(info: zipfile.ZipInfo, dest_root: Path) -> None:
    target = (dest_root / info.filename).resolve()
    if dest_root.resolve() not in target.parents:
        raise SystemExit(f"refusing unsafe path in zip: {info.filename}")
    if target.exists() and target.stat().st_size == info.file_size:
        print(f"present  {info.filename}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            started = time.time()
            with _get(info.header_offset, info.header_offset + 29) as response:
                header = response.read()
            if header[:4] != b"PK\x03\x04":
                raise OSError("bad local header")
            name_len, extra_len = struct.unpack("<HH", header[26:30])
            begin = info.header_offset + 30 + name_len + extra_len
            decoder = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
            crc, written = 0, 0
            partial = target.with_name(target.name + ".part")
            with _get(begin, begin + info.compress_size - 1, timeout=300) as response, partial.open("wb") as handle:
                while True:
                    chunk = response.read(4 << 20)
                    if not chunk:
                        break
                    out = decoder.decompress(chunk) if decoder else chunk
                    crc, written = zlib.crc32(out, crc), written + len(out)
                    handle.write(out)
                if decoder:
                    out = decoder.flush()
                    crc, written = zlib.crc32(out, crc), written + len(out)
                    handle.write(out)
            if written != info.file_size or (crc & 0xFFFFFFFF) != info.CRC:
                raise OSError(f"check failed: {written}/{info.file_size} bytes, crc {crc & 0xFFFFFFFF:08x}/{info.CRC:08x}")
            partial.replace(target)
            print(f"ok       {info.filename} ({info.file_size / 1e6:.1f} MB, {time.time() - started:.0f} s)")
            return
        except OSError as error:
            print(f"retry {attempt + 1}: {info.filename}: {error}")
            time.sleep(5)
    raise SystemExit(f"failed to download {info.filename}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default="data/uwb_falltrack", help="folder that receives abc2027_uwb_train_v1/ (default: %(default)s)")
    parser.add_argument("--subset", choices=["all", "smoke"], default="all")
    parser.add_argument("--list", action="store_true", help="only list the members that would be downloaded")
    args = parser.parse_args(argv)
    size = remote_size()
    archive = zipfile.ZipFile(io.BufferedReader(RangeFile(size), buffer_size=1 << 20))
    members = [m for m in archive.infolist() if wanted(m.filename, args.subset)]
    members.sort(key=lambda m: ("/streams/" in m.filename, m.filename))
    print(f"{len(members)} files, {sum(m.compress_size for m in members) / 1e9:.2f} GB to transfer, {sum(m.file_size for m in members) / 1e9:.2f} GB on disk")
    if args.list:
        for member in members:
            print(f"{member.file_size:>14,}  {member.filename}")
        return 0
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for member in members:
        extract(member, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
