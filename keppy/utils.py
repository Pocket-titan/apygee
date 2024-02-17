def shorten_fstring_number(x: str):
    x = x.lower()

    if "e+" in x or "e-" in x:
        [l, r] = x.split("e")
        l = l.rstrip("0").rstrip(".")
        r = r[0] + r[1:].lstrip("0")
        return l + "e" + r if len(r) > 1 else l

    return x.rstrip("0").rstrip(".")
