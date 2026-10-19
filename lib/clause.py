"""S588x: the relation window of a source ends at its clause -- the first token that carries a line break or starts with
sentence-final punctuation (". ? !"). A reverse modifier is a post-nominal phrase of the referring NP, so it lies in the
same clause / option line as the source; tokens past the boundary belong to the next sentence or option."""


def clause_len(tok, ids, pos, w):
    """Number of tokens after position pos (at most w) before the clause boundary."""
    n = 0
    for q in range(pos + 1, min(pos + 1 + w, len(ids))):
        s = tok.decode([ids[q]])
        st = s.strip()
        if "\n" in s or (st and st[0] in ".?!"):
            break
        n += 1
    return n
