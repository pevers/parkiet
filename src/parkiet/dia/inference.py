from parkiet.dia.model import Dia
import torch

torch.manual_seed(0)

model = Dia.from_local(
    config_path="config.json",
    checkpoint_path="weights/dia-nl-v3.pth",
    compute_dtype="float32",
    #   device=torch.device("cpu")
)

# text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
# text = "[S1] wie is Peter Evers? [S2] ja goede vraag, dat is één van de grote denkers van onze tijd. [S1] oh ja? [S2]. jazeker, Aristoteles, Plato en Peter Evers (laughs)."
# text = "[S1] een andere tekst is ook wel eens fijn. [S2] ja toch, de hele tijd maar dat gepraat over Peter Evers. [S1] om moe van te worden ja."
text = "[S1] hoera voor de koning. [S2] hoera (laughs) hoera (laughs) hoera."

for i in range(0, 10):
    output = model.generate(
        text,
        use_torch_compile=False,
        verbose=True,
        cfg_scale=3.0,
        temperature=1.8,
        top_p=0.90,
        cfg_filter_top_k=50,
        max_tokens=800,
    )
    model.save_audio(f"test_3_{i}.mp3", output)
