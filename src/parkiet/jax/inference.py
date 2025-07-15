from parkiet.jax.model import Dia
from pathlib import Path

checkpoint_path = (Path("weights") / "jax-v1").resolve()
model = Dia.from_local(
    config_path="config.json",
    checkpoint_path=checkpoint_path.as_posix(),
    compute_dtype="float32",
)

text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

output = model.generate(
    text,
    verbose=True,
    cfg_scale=3.0,
    temperature=1.8,
    top_p=0.90,
    cfg_filter_top_k=50,
)

model.save_audio("simple.mp3", output)
