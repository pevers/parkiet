from parkiet.dia.model import Dia
import torch


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    text = [
        "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face.",
    ]
    model = Dia.from_pretrained(
        "nari-labs/Dia-1.6B-0626", compute_dtype="float16", device=device
    )
    audio = model.generate(
        text=text,
        use_torch_compile=False,
        verbose=True,
        cfg_scale=3.0,
        temperature=1.8,
        top_p=0.90,
        cfg_filter_top_k=50,
    )
    model.save_audio("output.wav", audio)


if __name__ == "__main__":
    main()
