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
    model = Dia.from_local(
        config_path="config.json",
        checkpoint_path="weights/dia-v0_1-epoch100.pth",
        compute_dtype=ComputeDtype.BFLOAT16,
        device=device,
    )
    audio = model.generate(
        text="[S1] in Alkmaar. [S2] school. daar lopen we tegenaan zo. dat is de Sint-Mattheusschool. (laughs). [S1] hier woont Dave eind jaren zeventig begin jaren tachtig samen met zijn vader Dirk zijn vijf jaar jongere broertje Roy en zijn moeder Jos. Jos is anders dan andere moeders. [S2] je zag natuurlijk dat andere kinderen dat hun ouders wel naar ouderenavonden gingen. en uh dat hoorde je ook. en mijn",
        temperature=0.0,
        cfg_scale=1.0,
    )
    model.save_audio("output.wav", audio)


if __name__ == "__main__":
    main()
