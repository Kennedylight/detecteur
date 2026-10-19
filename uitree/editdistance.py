"""Distance d'édition (Levenshtein), pour les textes (CER) comme pour les suites de types d'enfants."""
from collections.abc import Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Nombre minimal d'insertions, suppressions, substitutions pour passer de a à b."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
