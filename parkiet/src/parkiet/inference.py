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
    text = "[S1] Honestly, I could use another coffee right now."
    model = Dia.from_local(
        config_path="config.json",
        checkpoint_path="weights/dia-v0_1.pth",
        device=device,
    )
    audio = model.generate(text=text)
    model.save_audio("output.wav", audio)


if __name__ == "__main__":
    main()
