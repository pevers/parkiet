from parkiet.dia.model import ComputeDtype, Dia
import torch


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
    model = Dia.from_local(
        config_path="config.json",
        checkpoint_path="weights/dia-v0_1.pth",
        compute_dtype=ComputeDtype.BFLOAT16,
        device=device,
    )
    audio = model.generate(text=text)
    model.save_audio("output.wav", audio)


if __name__ == "__main__":
    main()
