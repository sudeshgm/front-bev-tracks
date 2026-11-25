import urllib.request, struct, zlib, io
class RZip:
    def __init__(self, url):
        self.url = url
        self.size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30).headers["Content-Length"])
        tail = self.get(max(0, self.size - 65536 - 22), self.size - 1)
        i = tail.rfind(b"PK\x05\x06"); cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
        if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:  # zip64
            j = tail.rfind(b"PK\x06\x06"); cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
        cd = self.get(cd_off, cd_off + cd_size - 1); self.ent = {}; p = 0
        while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
            meth, = struct.unpack("<H", cd[p + 10:p + 12]); csz, usz = struct.unpack("<II", cd[p + 20:p + 28])
            nl, el, kl = struct.unpack("<HHH", cd[p + 28:p + 34]); off, = struct.unpack("<I", cd[p + 42:p + 46])
            name = cd[p + 46:p + 46 + nl].decode(); ex = cd[p + 46 + nl:p + 46 + nl + el]
            vals = []; q = 0
            while q < len(ex):
                hid, hl = struct.unpack("<HH", ex[q:q + 4])
                if hid == 1: vals = list(struct.unpack("<" + "Q" * (hl // 8), ex[q + 4:q + 4 + hl]))
                q += 4 + hl
            if usz == 0xFFFFFFFF: usz = vals.pop(0)
            if csz == 0xFFFFFFFF: csz = vals.pop(0)
            if off == 0xFFFFFFFF: off = vals.pop(0)
            self.ent[name] = (meth, csz, off); p += 46 + nl + el + kl
    def get(self, a, b):
        import time
        for t in range(6):
            try:
                r = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})
                return urllib.request.urlopen(r, timeout=60).read()
            except Exception:
                if t == 5: raise
                time.sleep(2 + 3 * t)
    def read(self, name):
        meth, csz, off = self.ent[name]
        h = self.get(off, off + 29); nl, el = struct.unpack("<HH", h[26:30])
        d = self.get(off + 30 + nl + el, off + 30 + nl + el + csz - 1)
        return d if meth == 0 else zlib.decompress(d, -15)
