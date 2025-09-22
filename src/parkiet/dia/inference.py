from parkiet.dia.model import Dia
import torch

torch.manual_seed(0)

model = Dia.from_local(
    config_path="config.json",
    checkpoint_path="weights/dia-nl-v1.pth",
    compute_dtype="float32",
    # device=torch.device("cpu")
)

# text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
text = "[S1] denk je dat je een open source model kan trainen met weinig geld en middelen? [S2] ja ik denk het wel. [S1] oh ja, hoe dan? [S2] nou kijk maar in de repo (laughs)."

output = model.generate(
    text,
    use_torch_compile=False,
    verbose=True,
    cfg_scale=3.0,
    temperature=1.8,
    top_p=0.90,
    cfg_filter_top_k=50
)
model.save_audio(f"example.mp3", output)
