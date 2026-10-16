"""Fabrique une fausse maquette 'Contacts' (fond blanc) pour tester notre chaîne.

Écrit aussi la VÉRITÉ TERRAIN (maquette_test_gt.json) : pour chaque élément dessiné,
sa boîte exacte et son type attendu. La boîte est mesurée sur les pixels : on compare
l'image avant / après le dessin de l'élément, la zone modifiée est sa boîte.
"""
import random
from collections.abc import Callable

from PIL import Image, ImageChops, ImageDraw, ImageFont

from uitree.groundtruth import GroundTruth, save_ground_truth

W, H = 400, 270


class _Recorder:
    """Dessine sur l'image et note la boîte réelle (en pixels) de chaque élément."""

    def __init__(self, img: Image.Image) -> None:
        self.img = img
        self.draw = ImageDraw.Draw(img)
        self.truth: list[GroundTruth] = []

    def add(self, name: str, kind: str, paint: Callable[[ImageDraw.ImageDraw], None],
            text: str | None = None) -> None:
        before = self.img.copy()
        paint(self.draw)
        x0, y0, x1, y1 = ImageChops.difference(self.img, before).getbbox()
        self.truth.append(GroundTruth(name, (x0, y0, x1 - x0, y1 - y0), kind, text))


def make_test_image() -> tuple[Image.Image, list[GroundTruth]]:
    random.seed(1)
    img = Image.new("RGB", (W, H), "white")
    rec = _Recorder(img)
    try:
        big = ImageFont.load_default(size=16)
        small = ImageFont.load_default(size=11)
    except TypeError:                      # très vieilles versions de Pillow
        big = small = ImageFont.load_default()

    # En-tête : icône menu, titre, icône recherche
    def menu(d: ImageDraw.ImageDraw) -> None:
        d.rectangle([12, 12, 38, 18], fill="black")
        d.rectangle([12, 22, 38, 28], fill="black")
    rec.add("menu", "Icône", menu)
    rec.add("titre", "Texte", lambda d: d.text((150, 12), "Contacts", fill="black", font=big),
            "Contacts")
    rec.add("recherche", "Icône",
            lambda d: d.ellipse([360, 10, 384, 34], outline="black", width=3))

    names = [("Peter Limonez", "The road to success is always under construction"),
             ("Sofia Morell", "My silence is just another word for pain"),
             ("Alan Leonardo", "I am not failed my success is postponed")]
    for i, (name, msg) in enumerate(names):
        y = 60 + i * 70

        def avatar(d: ImageDraw.ImageDraw, y: int = y) -> None:
            # disque avec des couleurs aléatoires (simule une photo)
            for _ in range(300):
                px, py = random.randint(10, 58), random.randint(y, y + 48)
                if (px - 34) ** 2 + (py - (y + 24)) ** 2 < 24 ** 2:
                    d.point((px, py), fill=tuple(random.randint(0, 255) for _ in range(3)))
            d.ellipse([10, y, 60, y + 50], outline=(60, 120, 60), width=2)
        rec.add(f"avatar{i + 1}", "Image", avatar)
        rec.add(f"nom{i + 1}", "Texte",
                lambda d, y=y, t=name: d.text((75, y + 2), t, fill="black", font=big), name)
        rec.add(f"message{i + 1}", "Texte",
                lambda d, y=y, t=msg: d.text((75, y + 30), t, fill=(130, 130, 130), font=small),
                msg)
        rec.add(f"badge{i + 1}", "Composant",
                lambda d, y=y: d.rounded_rectangle([335, y + 26, 385, y + 44], radius=4,
                                                   fill=(110, 200, 100)))
        rec.add(f"badge{i + 1}_texte", "Texte",
                lambda d, y=y: d.text((343, y + 29), "MOBILE", fill="white", font=small),
                "MOBILE")
    return img, rec.truth


def make_form_image() -> tuple[Image.Image, list[GroundTruth]]:
    """Formulaire de connexion : Conteneur, Champ, Bouton (plein et contour), Icône, Image, badge.

    Conçu pour mesurer les TYPES ; mêmes règles de vérité terrain (boîte = pixels modifiés)."""
    random.seed(2)
    img = Image.new("RGB", (420, 360), (243, 244, 248))
    rec = _Recorder(img)
    big, small = ImageFont.load_default(size=18), ImageFont.load_default(size=12)
    blue = (37, 99, 235)

    def text(name: str, xy: tuple[int, int], s: str, fill: tuple[int, int, int], font) -> None:
        rec.add(name, "Texte", lambda d: d.text(xy, s, fill=fill, font=font), s)

    rec.add("carte", "Conteneur", lambda d: d.rounded_rectangle(
        [10, 10, 409, 349], radius=12, fill="white", outline=(225, 227, 232)))
    text("titre", (30, 26), "Connexion", (17, 24, 39), big)

    def photo(d: ImageDraw.ImageDraw) -> None:
        for py in range(22, 82, 3):
            for px in range(330, 390, 3):
                d.rectangle([px, py, px + 2, py + 2], fill=tuple(random.randint(0, 220) for _ in range(3)))
    rec.add("photo", "Image", photo)

    text("label_email", (30, 70), "Email", (55, 65, 81), small)
    rec.add("champ_email", "Champ", lambda d: d.rounded_rectangle(
        [30, 88, 389, 124], radius=6, fill="white", outline=(200, 203, 210)))
    text("placeholder_email", (42, 99), "nom@exemple.com", (150, 150, 150), small)
    rec.add("champ_recherche", "Champ", lambda d: d.rounded_rectangle(
        [30, 140, 389, 176], radius=6, fill="white", outline=(200, 203, 210)))

    def loupe(d: ImageDraw.ImageDraw) -> None:
        d.ellipse([42, 150, 54, 162], outline=(100, 100, 100), width=2)
        d.line([52, 160, 58, 166], fill=(100, 100, 100), width=2)
    rec.add("icone_loupe", "Icône", loupe)
    text("placeholder_recherche", (66, 151), "Rechercher...", (150, 150, 150), small)

    rec.add("bouton_principal", "Bouton", lambda d: d.rounded_rectangle(
        [30, 196, 200, 236], radius=8, fill=blue))
    text("bouton_principal_texte", (77, 209), "Se connecter", "white", small)
    rec.add("bouton_secondaire", "Bouton", lambda d: d.rounded_rectangle(
        [220, 196, 389, 236], radius=8, fill="white", outline=blue, width=2))
    text("bouton_secondaire_texte", (281, 209), "Annuler", blue, small)

    rec.add("icone_lecture", "Icône", lambda d: d.polygon([(32, 258), (32, 280), (52, 269)], fill=(16, 185, 129)))
    rec.add("icone_rond", "Icône", lambda d: d.ellipse([70, 258, 92, 280], fill=(245, 158, 11)))
    rec.add("icone_carre", "Icône", lambda d: d.rectangle([110, 259, 130, 279], outline=(55, 65, 81), width=2))
    rec.add("badge", "Composant", lambda d: d.rounded_rectangle(
        [300, 262, 370, 282], radius=10, fill=(254, 226, 226)))
    text("badge_texte", (311, 266), "Nouveau", (185, 28, 28), small)
    text("lien", (30, 300), "Mot de passe oublié ?", blue, small)

    def barre(d: ImageDraw.ImageDraw) -> None:
        d.rounded_rectangle([30, 330, 389, 338], radius=4, fill=(229, 231, 235))
        d.rounded_rectangle([30, 330, 250, 338], radius=4, fill=(34, 197, 94))
    rec.add("barre", "Composant", barre)
    return img, rec.truth


SCENES = {"maquette_test": make_test_image, "maquette_formulaire": make_form_image}


if __name__ == "__main__":
    for stem, make in SCENES.items():
        image, truth = make()
        image.save(f"{stem}.png")
        save_ground_truth(truth, f"{stem}_gt.json")
        print(f"{stem}.png créée, {len(truth)} éléments de vérité terrain dans {stem}_gt.json")
