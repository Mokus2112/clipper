"""YAMNet (TFLite, ~4 MB, CPU) audio tagging: reaction sounds (laughter, screaming, cheering) and music."""

import numpy as np

from .common import ROOT

MODEL = ROOT / "models" / "yamnet.tflite"
WINDOW = 15600  # 0.975 s at 16 kHz, YAMNet's fixed input size
REACTION = [11, 13, 15, 18, 61, 62]  # Screaming, Laughter, Giggle, Chuckle, Cheering, Applause
MUSIC = 132


class Yamnet:
    def __init__(self, threads: int = 4):
        from ai_edge_litert.interpreter import Interpreter

        self.it = Interpreter(str(MODEL), num_threads=threads)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]

    def tag(self, second: np.ndarray) -> tuple[float, float]:
        """(reaction, music) scores for one second of 16 kHz mono float audio."""
        x = np.zeros(WINDOW, dtype=np.float32)
        x[: min(WINDOW, len(second))] = second[:WINDOW]
        self.it.set_tensor(self.inp["index"], x.reshape(self.inp["shape"]))
        self.it.invoke()
        scores = self.it.get_tensor(self.out["index"]).reshape(-1)
        return float(scores[REACTION].max()), float(scores[MUSIC])
