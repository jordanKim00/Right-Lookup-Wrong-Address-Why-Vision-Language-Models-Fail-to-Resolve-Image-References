"""The reference-expression inventory, and the splits that make it a generalisation test.

Where the budget goes is not a matter of taste here; it was measured.  The trained compiler
routes perfectly on domains it never saw (phone screenshots, recipe steps), on image counts it
never saw (2 through 10 from a single N = 6), and on content it never saw -- and it fails,
deterministically, the moment the wording changes.  "the fourth image from the end" resolves
1.000 of the time; drop three words to "the fourth image" and it drops to .224; say "the last
image" and it lands exactly two images early on all 133 items, because that phrase only ever
appeared in training as the starting point of a numeric move.

So images are the cheap axis and sentences are the expensive one, which is convenient, because
sentences cost nothing to write.  This module is the inventory: for every way of naming an
image that we intend to support, a surface form and the typed address it denotes, plus the
splits that hold some forms out entirely so that generalisation is tested rather than assumed.

Each entry returns the question text, the phrase whose final token is the injection site, and
the (origin, direction, distance) the phrase denotes.  A self-test walks every form at every
length and checks the declared address actually transports to the intended image, so a wrong
entry is a failed assertion rather than a silently mislabelled training example.
"""
from __future__ import annotations
import itertools, json, random, sys

ORIGIN_FRONT, ORIGIN_BACK, ORIGIN_VISUAL = 0, 1, 2
DIR_MINUS, DIR_PLUS = 0, 1
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh",
       "eighth", "ninth", "tenth"]
NTH = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]
CARD = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


# --- each form: (family, name, builder).  builder(n, target) -> (phrase, origin, dir, dist)
# or None when the form cannot express that target at that length.

def _front(n, t, phrase):
    return phrase, ORIGIN_FRONT, DIR_PLUS, t


def _back(n, t, phrase):
    return phrase, ORIGIN_BACK, DIR_MINUS, n - 1 - t


FORMS = [
    # forward ordinal -- the trained form and four paraphrases of it
    ("fwd_ordinal", "canonical",
     lambda n, t: _front(n, t, f"the {ORD[t]} image from the beginning") if t < 10 else None),
    ("fwd_ordinal", "bare",
     lambda n, t: _front(n, t, f"the {ORD[t]} image") if t < 10 else None),
    ("fwd_ordinal", "digit",
     lambda n, t: _front(n, t, f"the {NTH[t]} image") if t < 10 else None),
    ("fwd_ordinal", "numbered",
     lambda n, t: _front(n, t, f"image {t + 1}") if t < 10 else None),
    ("fwd_ordinal", "pronominal",
     lambda n, t: _front(n, t, f"the {ORD[t]} one") if t < 10 else None),
    ("fwd_ordinal", "counted",
     lambda n, t: _front(n, t, f"the {ORD[t]} picture in the sequence") if t < 10 else None),

    # backward ordinal -- the trained form and its paraphrases
    ("bwd_ordinal", "canonical",
     lambda n, t: _back(n, t, f"the {ORD[n - 1 - t]} image from the end") if n - 1 - t < 10 else None),
    ("bwd_ordinal", "from_last",
     lambda n, t: _back(n, t, f"the {ORD[n - 1 - t]} image from the last") if n - 1 - t < 10 else None),
    ("bwd_ordinal", "to_last",
     lambda n, t: _back(n, t, f"the {ORD[n - 1 - t]}-to-last image") if 0 < n - 1 - t < 10 else None),
    ("bwd_ordinal", "counting_back",
     lambda n, t: _back(n, t, f"the {ORD[n - 1 - t]} image counting back") if n - 1 - t < 10 else None),

    # unary boundary -- the single most common way people name an edge, and absent from
    # training, which is why "the last image" routed two images early on every GUI item
    ("boundary", "last",
     lambda n, t: _back(n, t, "the last image") if t == n - 1 else None),
    ("boundary", "final",
     lambda n, t: _back(n, t, "the final image") if t == n - 1 else None),
    ("boundary", "closing",
     lambda n, t: _back(n, t, "the very last image") if t == n - 1 else None),
    ("boundary", "first",
     lambda n, t: _front(n, t, "the first image") if t == 0 else None),
    ("boundary", "leading",
     lambda n, t: _front(n, t, "the very first image") if t == 0 else None),
    ("boundary", "opening",
     lambda n, t: _front(n, t, "the opening image") if t == 0 else None),

    # numeric move from an edge -- distance carried by a digit rather than an ordinal
    ("numeric_move", "front_toward_end",
     lambda n, t: (f"starting at the first image, move {t} places toward the end",
                   ORIGIN_FRONT, DIR_PLUS, t) if 0 < t < 10 else None),
    ("numeric_move", "back_toward_start",
     lambda n, t: (f"starting at the last image, move {n - 1 - t} places toward the beginning",
                   ORIGIN_BACK, DIR_MINUS, n - 1 - t) if 0 < n - 1 - t < 10 else None),
    ("numeric_move", "front_word",
     lambda n, t: (f"starting at the first image, go {CARD[t]} images forward",
                   ORIGIN_FRONT, DIR_PLUS, t) if 0 < t < 10 else None),
    ("numeric_move", "back_word",
     lambda n, t: (f"starting at the last image, go {CARD[n - 1 - t]} images back",
                   ORIGIN_BACK, DIR_MINUS, n - 1 - t) if 0 < n - 1 - t < 10 else None),
]

# Whether a form can stand where a noun phrase stands.  The numeric moves are clauses --
# "starting at the first image, move 4 places toward the end" -- and dropping one into a
# noun-phrase slot produces "What object is in starting at the first image, ...?", which is
# not a sentence.  A first version of the manifest did exactly that, and the router answered
# with high confidence and the wrong image on every one of them.
SLOT = {"fwd_ordinal": "np", "bwd_ordinal": "np", "boundary": "np",
        "relational": "np", "numeric_move": "clause"}


# Relational forms need an anchor image, so they are built separately against a label.
REL_FORMS = [
    ("relational", "immediately_after",
     lambda lab: (f"the image immediately after the image containing a {lab}", DIR_PLUS, 1)),
    ("relational", "right_after",
     lambda lab: (f"the image right after the one showing a {lab}", DIR_PLUS, 1)),
    ("relational", "immediately_before",
     lambda lab: (f"the image immediately before the image containing a {lab}", DIR_MINUS, 1)),
    ("relational", "just_before",
     lambda lab: (f"the image just before the one showing a {lab}", DIR_MINUS, 1)),
]

# --- answer forms.  Training asked one question -- name the object -- so the value pathway
# only ever had to carry a category.  On q_bench+ the injected vector's effect correlated
# +0.747 with its own negation, meaning what reached the decision barely depended on which
# image was routed.  Varying what the answer demands is how the value pathway gets a reason
# to carry the source rather than a shortcut to one output distribution.
ANSWER_FORMS = ["name", "yesno_present", "choose", "yesno_compare", "relation_order"]


def build_question(kind, phrases, targets, labels, rng):
    """Turn one or two resolved addresses into a question and its answer.

    The five kinds ask progressively more of the routed value.  ``name`` needs only a
    category, which is all the current training ever required -- and a value that carries only
    a category is exactly what the q_bench+ diagnosis found: injecting it moved the decision
    almost identically whether the right image or the wrong one was routed.  ``yesno_present``
    makes the category compete with a probe word, so a value that merely lifts every plausible
    noun cannot win.  ``choose`` puts three distractor categories in the prompt, so the value
    has to separate them at the readout rather than shift a whole family of tokens.  The last
    two take two addresses in one sentence, which a single-address design cannot express at
    all: ``yesno_compare`` needs the two routed values to be distinguishable from each other,
    and ``relation_order`` needs the position with the content thrown away.

    Returns (question, answer) or None when the ingredients do not allow the kind.
    """
    vocab = sorted(set(labels))
    if kind in ("name", "yesno_present", "choose"):
        t = targets[0]
        if kind == "name":
            return f"What object is in {phrases[0]}?", labels[t]
        if kind == "yesno_present":
            yes = rng.random() < 0.5
            probe = labels[t] if yes else rng.choice([w for w in vocab if w != labels[t]])
            return f"Is there a {probe} in {phrases[0]}?", "yes" if yes else "no"
        others = [w for w in vocab if w != labels[t]]
        if len(others) < 3: return None
        picks = [labels[t]] + rng.sample(others, 3)
        rng.shuffle(picks)
        listing = " ".join(f"{L}) {w}" for L, w in zip("ABCD", picks))
        return (f"Which of these is in {phrases[0]}? {listing}",
                "ABCD"[picks.index(labels[t])])
    if len(phrases) < 2 or len(targets) < 2 or targets[0] == targets[1]: return None
    a, b = targets
    if kind == "yesno_compare":
        return (f"Do {phrases[0]} and {phrases[1]} show the same object?",
                "yes" if labels[a] == labels[b] else "no")
    if kind == "relation_order":
        return (f"Does {phrases[0]} come before {phrases[1]}?",
                "yes" if a < b else "no")
    return None


# Which surface token inside a phrase carries the distance.  The parser reads the distance
# from that token's hidden state, not from the phrase as a whole, so getting this wrong makes
# the router read the wrong constituent and look broken when it is fine -- which is exactly
# what a first version of the evaluation harness did.  Boundary words and relational phrases
# have no distance constituent at all; training falls back to the query position for those.
def distance_word(family, form, n, target):
    if family == "fwd_ordinal":
        return {"digit": NTH[target], "numbered": str(target + 1)}.get(form, ORD[target])
    if family == "bwd_ordinal":
        return ORD[n - 1 - target]
    if family == "numeric_move":
        d = target if form.startswith("front") else n - 1 - target
        return CARD[d] if form.endswith("word") else str(d)
    return None


def denote(origin, direction, distance, n, anchor=None):
    """Where a typed address lands, or None if it leaves the graph."""
    start = 0 if origin == ORIGIN_FRONT else (n - 1 if origin == ORIGIN_BACK else anchor)
    if start is None: return None
    step = 1 if direction == DIR_PLUS else -1
    landing = start + step * distance
    return landing if 0 <= landing < n else None


def enumerate_forms(lengths=range(2, 11)):
    """Every (form, n, target) the inventory can express, with its denotation checked."""
    out = []
    for n in lengths:
        for t in range(n):
            for family, name, build in FORMS:
                got = build(n, t)
                if got is None: continue
                phrase, origin, direction, distance = got
                if distance > 9: continue
                landing = denote(origin, direction, distance, n)
                assert landing == t, (family, name, n, t, landing)
                out.append(dict(family=family, form=name, n=n, target=t, phrase=phrase,
                                origin=origin, direction=direction, distance=distance))
    return out


HELD_OUT_FORMS = {
    # one paraphrase per family is never trained, so the split measures transfer to a new
    # way of saying an address rather than to a new address
    "fwd_ordinal": "pronominal",
    "bwd_ordinal": "to_last",
    "boundary": "closing",
    "numeric_move": "back_word",
    "relational": "just_before",
}


def split_of(row, held_out_n=(9, 10)):
    """Train, or the reason it is held out.  Length and wording are held out separately so a
    failure says which one it was."""
    if row["form"] == HELD_OUT_FORMS.get(row["family"]): return "heldout_form"
    if row["n"] in held_out_n: return "heldout_length"
    return "train"


def main():
    rows = enumerate_forms()
    import collections
    print(f"{len(rows)} (form, N, target) combinations, all denotations verified")
    by = collections.Counter((r["family"], r["form"]) for r in rows)
    for (fam, form), c in sorted(by.items()):
        mark = "  HELD OUT" if HELD_OUT_FORMS.get(fam) == form else ""
        print(f"  {fam:14s} {form:20s} {c:4d}{mark}")
    sp = collections.Counter(split_of(r) for r in rows)
    print("\nsplits:", dict(sp))
    print("\nsample phrases at N=7, target=4:")
    for r in rows:
        if r["n"] == 7 and r["target"] == 4:
            print(f"  {r['family']:14s} {r['form']:20s} "
                  f"origin={r['origin']} dir={r['direction']} d={r['distance']}  \"{r['phrase']}\"")
    rng = random.Random(947)
    labels = ["dog", "truck", "church", "ball", "saw", "horn", "pump"]
    print("\nanswer forms, N=7 with two addresses resolved to images 4 and 1:")
    two = [r for r in rows if r["n"] == 7 and r["form"] == "canonical"
           and r["family"] == "fwd_ordinal" and r["target"] in (4, 1)]
    ph = [r["phrase"] for r in sorted(two, key=lambda r: -r["target"])]
    for kind in ANSWER_FORMS:
        got = build_question(kind, ph, [4, 1], labels, rng)
        print(f"  {kind:16s} {got[0] if got else '-'}")
        if got: print(f"  {'':16s}   -> {got[1]}")
    print(f"sentence budget vs current training (3 forms, N=6 only): "
          f"{len(rows)} vs {sum(1 for r in rows if r['n'] == 6 and r['form'] == 'canonical')}"
          f" comparable canonical rows")


if __name__ == "__main__":
    main()
