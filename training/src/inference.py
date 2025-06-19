from dia.model import Dia
import torch


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = Dia.from_local(
        config_path="config.json",
        checkpoint_path="weights/dia-v0_1.pth",
        device=device,
    )
    print(model.model)
    audio = model.generate(
        text="[S1] Howdy sir, how are you doing! [S2] I'm doing well, thank you for asking, please tell me about your day.",
    )
    model.save_audio("output.wav", audio)


if __name__ == "__main__":
    main()
