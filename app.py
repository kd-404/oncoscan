"""Gradio app: upload a histopathology image, get a malignancy screen + Grad-CAM."""
import argparse

import gradio as gr

from oncoscan.predict import Predictor

DISCLAIMER = ("**Research/portfolio demo only — not a medical device and not for diagnosis.** "
              "The model is tuned for high recall (few missed cancers), so false alarms are expected.")


def build_app(ckpt):
    predictor = Predictor(ckpt)

    def run(img):
        if img is None:
            return None, None, ""
        r = predictor.predict(img)
        label = "Suspicious – refer for pathologist review" if r["malignant"] else "No malignancy flagged"
        return {"Malignant": r["p_malignant"], "Benign": 1 - r["p_malignant"]}, r["cam"], label

    with gr.Blocks(title="OncoScan") as demo:
        gr.Markdown("# 🔬 OncoScan — early-stage cancer cell screening\n" + DISCLAIMER)
        with gr.Row():
            inp = gr.Image(type="pil", label="Histopathology image")
            with gr.Column():
                probs = gr.Label(label="Probability")
                verdict = gr.Textbox(label="Screening result")
                cam = gr.Image(label="Grad-CAM (regions driving the prediction)")
        gr.Button("Analyze", variant="primary").click(run, inp, [probs, cam, verdict])
        gr.Markdown(f"Decision threshold (recall-tuned): `{predictor.threshold:.3f}`")
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/oncoscan.pt")
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    build_app(a.ckpt).launch(share=a.share)
