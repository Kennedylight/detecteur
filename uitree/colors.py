"""Mesures de couleur sur les pixels (BVR, comme OpenCV)."""
import numpy as np


def dominant_color(pixels: np.ndarray) -> np.ndarray:
    """Couleur la plus fréquente : on regroupe par pas de 16 (contre le bruit), puis on prend la
    MOYENNE réelle des pixels du groupe gagnant (le centre du groupe peut être à 7 près par canal,
    ce qui suffit à faire passer un fond uni pour une surface avec un seuil bas)."""
    flat = pixels.reshape(-1, 3)
    q = flat // 16
    _, inverse, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    return np.rint(flat[inverse.ravel() == counts.argmax()].mean(axis=0)).astype(np.int16)


def color_distance(pixels: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Somme |ΔB|+|ΔV|+|ΔR| de chaque pixel à color."""
    return np.abs(pixels.astype(np.int16) - color).sum(axis=-1)


def palette_ratio(pixels: np.ndarray, k: int = 6, step: int = 32) -> float:
    """Part des pixels couverts par les k couleurs les plus fréquentes (quantifiées par pas de step).

    Mesuré au centre (center_crop 0,6) : cartes, tuiles, bannières d'interface 0,91 à 0,96 (quelques
    aplats, même en dégradé) ; photos d'avatars 0,30 à 0,77. Distingue mieux qu'une seule couleur
    dominante : une carte remplie de tuiles colorées n'a pas de couleur majoritaire, mais peu de
    couleurs."""
    _, counts = np.unique((pixels // step).reshape(-1, 3), axis=0, return_counts=True)
    return float(np.sort(counts)[::-1][:k].sum() / counts.sum())


def center_crop(crop: np.ndarray, fraction: float) -> np.ndarray:
    """Partie centrale (fraction de la largeur et de la hauteur) : les coins d'un avatar rond sont
    du fond uni, qui fausse les mesures faites sur toute la boîte."""
    h, w = crop.shape[:2]
    my, mx = int(h * (1 - fraction) / 2), int(w * (1 - fraction) / 2)
    return crop[my:h - my, mx:w - mx]


def count_colors(pixels: np.ndarray, step: int = 32) -> int:
    """Nombre de couleurs distinctes après quantification (pixels : N x 3 ou H x W x 3)."""
    return len(np.unique((pixels // step).reshape(-1, 3), axis=0))


def to_hex(bgr: np.ndarray) -> str:
    b, g, r = (int(v) for v in bgr)
    return f"#{r:02x}{g:02x}{b:02x}"
