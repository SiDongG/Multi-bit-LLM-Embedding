import random
import numpy as np
import torch
import tkinter as tk
from tkinter import messagebox
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from Main.runner import MyEntropyHashWatermarkLogitsProcessor, NoWatermarkLogitsProcessor
from Main.watermarkutils import (
    construct_segments,
    allocate_bits_proportional_to_entropy,
)



def generate_watermarked_text(
    model,
    tokenizer,
    secret_key,
    P_X,
    N_tokens=200,
    bit_len=48,
    k=4,
    num_segments_total=15,
    prompt="Jen jumps into the water",
):
    """
    Generate one watermarked text using your entropy-hash watermark scheme.
    Returns (text, bitstream, generated_ids).
    """

    # 1) Random payload bits
    bitstream = "".join(random.choice("01") for _ in range(bit_len))

    # 2) Segment allocation from P_X
    segments_per_bin = construct_segments(P_X, num_segments_total)

    # 3) Allocate bits to segments
    segment_bits = allocate_bits_proportional_to_entropy(
        segments_per_bin,
        bitstream
    )

    # 4) Build watermark logits processor
    wm_processor = MyEntropyHashWatermarkLogitsProcessor(
        tokenizer=tokenizer,
        secret_key=secret_key,
        segments_per_bin=segments_per_bin,
        segment_bits=segment_bits,
        k=k,
        model=model,
    )
    processors = LogitsProcessorList([wm_processor])

    # 5) Generate text
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=N_tokens,
            logits_processor=processors,
            do_sample=True,
            top_p=0.9,
            temperature=1.0,
        )
    generated_ids = out[0]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return text, bitstream, generated_ids


def generate_unwatermarked_text(
    model,
    tokenizer,
    N_tokens=200,
    prompt="Jen jumps into the water",
):
    """
    Generate one unwatermarked (clean) text.
    You can either use a dummy NoWatermarkLogitsProcessor or no processor at all.
    """

    try:
        no_wm_processor = NoWatermarkLogitsProcessor()
        processors = LogitsProcessorList([no_wm_processor])
    except NameError:
        # Fallback: no logits_processor
        processors = None

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=N_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=1.0,
    )
    if processors is not None:
        gen_kwargs["logits_processor"] = processors

    with torch.no_grad():
        out = model.generate(**gen_kwargs)

    generated_ids = out[0]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, generated_ids


# =====================================================
# 3. GUI LOGIC
# =====================================================

class WatermarkTuringTestGUI:
    def __init__(self, master, items):
        """
        items: list of dicts with fields:
            - "text": str
            - "is_watermarked": bool
        """
        self.master = master
        self.items = items
        self.index = 0
        self.guesses = []

        master.title("Watermark Perception Test")

        # Text box
        self.text_widget = tk.Text(master, wrap="word", height=20, width=80)
        self.text_widget.pack(padx=10, pady=10)

        # Buttons frame
        btn_frame = tk.Frame(master)
        btn_frame.pack(pady=5)

        self.btn_wm = tk.Button(
            btn_frame, text="Watermarked", width=15,
            command=lambda: self.record_guess(True)
        )
        self.btn_nowm = tk.Button(
            btn_frame, text="Not Watermarked", width=15,
            command=lambda: self.record_guess(False)
        )
        self.btn_wm.grid(row=0, column=0, padx=5)
        self.btn_nowm.grid(row=0, column=1, padx=5)

        # Status label
        self.status_label = tk.Label(master, text="")
        self.status_label.pack(pady=5)

        self.show_current_item()

    def show_current_item(self):
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        if self.index < len(self.items):
            text = self.items[self.index]["text"]
            self.text_widget.insert(tk.END, text)
            self.text_widget.config(state=tk.DISABLED)
            self.status_label.config(
                text=f"Item {self.index+1} of {len(self.items)}"
            )
        else:
            # Done
            self.text_widget.insert(tk.END, "Experiment finished.")
            self.text_widget.config(state=tk.DISABLED)
            self.btn_wm.config(state=tk.DISABLED)
            self.btn_nowm.config(state=tk.DISABLED)
            self.show_results()

    def record_guess(self, guess_is_wm: bool):
        if self.index >= len(self.items):
            return

        item = self.items[self.index]
        self.guesses.append({
            "true": item["is_watermarked"],
            "guess": guess_is_wm
        })
        self.index += 1

        if self.index < len(self.items):
            self.show_current_item()
        else:
            self.show_current_item() 

    def show_results(self):
        # Compute accuracy and confusion matrix
        tp = fp = tn = fn = 0
        for g in self.guesses:
            t = g["true"]
            h = g["guess"]
            if t and h:
                tp += 1
            elif not t and h:
                fp += 1
            elif not t and not h:
                tn += 1
            elif t and not h:
                fn += 1

        total = len(self.guesses)
        acc = (tp + tn) / total if total > 0 else 0.0

        msg = (
            f"Results:\n\n"
            f"Total items: {total}\n"
            f"Accuracy: {acc:.3f}\n\n"
            f"Confusion matrix:\n"
            f"  True WM, guessed WM:     {tp}\n"
            f"  True WM, guessed not-WM: {fn}\n"
            f"  True not-WM, guessed WM: {fp}\n"
            f"  True not-WM, guessed not-WM: {tn}\n"
        )
        messagebox.showinfo("Results", msg)


# =====================================================
# 4. MAIN DRIVER
# =====================================================

def main():
    # ----------------- SETTINGS -----------------
    NUM_WM = 5
    NUM_NOWM = 5
    N_tokens = 200
    bit_len = 12
    k = 4
    secret_key = "my_super_secret_key"
    model_name = "gpt2"

    P_X = np.array([
        0.042, 0.02, 0.024, 0.04, 0.07,
        0.121, 0.139, 0.144, 0.14, 0.12, 0.068, 0.072
    ])

    # -------------- LOAD MODEL --------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    print("Generating watermarked and unwatermarked texts...")

    items = []

    # -------------- GENERATE WM TEXTS --------------
    for _ in range(NUM_WM):
        text, bitstream, _ = generate_watermarked_text(
            model=model,
            tokenizer=tokenizer,
            secret_key=secret_key,
            P_X=P_X,
            N_tokens=N_tokens,
            bit_len=bit_len,
            k=k,
        )
        items.append({
            "text": text,
            "is_watermarked": True,
            "bitstream": bitstream,
        })

    # -------------- GENERATE CLEAN TEXTS --------------
    for _ in range(NUM_NOWM):
        text, _ = generate_unwatermarked_text(
            model=model,
            tokenizer=tokenizer,
            N_tokens=N_tokens,
        )
        items.append({
            "text": text,
            "is_watermarked": False,
            "bitstream": None,
        })

    # -------------- SHUFFLE ITEMS --------------
    random.shuffle(items)

    print("Launching GUI...")
    root = tk.Tk()
    app = WatermarkTuringTestGUI(root, items)
    root.mainloop()


if __name__ == "__main__":
    main()
