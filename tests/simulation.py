"""Fait tourner la boucle de main.py sans webcam ni fenêtre, pour les tests."""
from functools import lru_cache

import main
from deformation import charger_portraits
from reperes import DetecteurReperes


class CameraFactice:
    """Remplace la webcam : renvoie les images données, puis répète la dernière."""

    def __init__(self, images):
        self.images = list(images)
        self.position = 0

    def read(self):
        image = self.images[min(self.position, len(self.images) - 1)]
        self.position += 1
        return True, image.copy()


@lru_cache(maxsize=1)
def portraits_de_test():
    return list(charger_portraits().values())


def lancer_boucle(monkeypatch, images, touches):
    """Lance la boucle et renvoie les images affichées, une par touche simulée."""
    touches = iter(touches)
    affichees = []
    monkeypatch.setattr(main.cv2, "imshow", lambda _, image: affichees.append(image.copy()))
    monkeypatch.setattr(main.cv2, "waitKey", lambda _: ord(next(touches)))
    monkeypatch.setattr(main.cv2, "getWindowProperty", lambda *_: 1.0)
    with DetecteurReperes(mode_video=True) as detecteur:
        main.executer(CameraFactice(images), detecteur, portraits_de_test())
    return affichees
