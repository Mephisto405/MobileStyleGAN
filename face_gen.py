import cv2

import os
import random_face.functional as F
from random_face.engine_openvino import EngineOpenvino
import argparse
import time

def get_face(engine: EngineOpenvino, cfg):
    start_time = time.time()
    face = engine.get_random_face(truncate=cfg["no_truncate"])
    end_time = time.time()
    elapsed_time = end_time - start_time
    return face, elapsed_time

def main(cfg):
    engine = EngineOpenvino(cfg)
    _ = engine.get_random_face(truncate=cfg["no_truncate"])

    wait_time = 1000

    face, elapsed_time = get_face(engine, cfg)
    win_name = f"Elapsed time: {elapsed_time}"
    cv2.namedWindow(win_name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(win_name, face)
    while cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) >= 1:
        keyCode = cv2.waitKey(wait_time)
        if (keyCode & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--style_dim", type=int, default=512)
    parser.add_argument("--n_samples", type=int, default=4096)
    parser.add_argument("--no_truncate", action="store_false")
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