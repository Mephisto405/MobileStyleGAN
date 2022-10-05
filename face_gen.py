import argparse
import os
import time

import cv2
import numpy as np
import random_face.functional as F
from random_face.engine_openvino import EngineOpenvino


def get_face(engine: EngineOpenvino, cfg):
    start_time = time.time()
    face = engine.get_random_face(truncate=cfg["no_truncate"])
    end_time = time.time()
    elapsed_time = end_time - start_time
    return face, elapsed_time


def main(cfg):
    engine = EngineOpenvino(cfg)
    face = engine.get_random_face(truncate=cfg["no_truncate"])

    if cfg["interactive"]:
        wait_time = 1000
        cv2.namedWindow("result", cv2.WINDOW_KEEPRATIO)

        while cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) >= 1:
            face, elapsed_time = get_face(engine, cfg)
            face = np.ascontiguousarray(face)
            win_name = f"Elapsed time: {elapsed_time}"
            cv2.putText(
                face,
                win_name,
                (0, int(face.shape[0] * 0.99)),
                cv2.FONT_HERSHEY_SIMPLEX,
                face.shape[0] / 1024,
                (255, 0, 0),
                1 + face.shape[0] // 512,
            )
            cv2.imshow("result", face)

            keyCode = cv2.waitKey(wait_time)
            if (keyCode & 0xFF) == ord("q"):
                cv2.destroyAllWindows()
                break
    else:
        cv2.imwrite("face.png", face)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--style_dim", type=int, default=512)
    parser.add_argument("--n_samples", type=int, default=4096)
    parser.add_argument("--no_truncate", action="store_false")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    cfg = vars(args)
    cfg["mnet_xml"] = os.path.join(cfg["dir"], "MappingNetwork.xml")
    cfg["mnet_bin"] = os.path.join(cfg["dir"], "MappingNetwork.bin")
    cfg["snet_xml"] = os.path.join(cfg["dir"], "SynthesisNetwork.xml")
    cfg["snet_bin"] = os.path.join(cfg["dir"], "SynthesisNetwork.bin")
    cfg["img_normalize"] = True
    cfg["img_range"] = [-1.0, 1.0]
    cfg["img_rgb2bgr"] = True

    main(cfg)
