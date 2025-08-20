from parkiet.jax.model import Dia
from pathlib import Path

# checkpoint_path = (Path("weights") / "jax-v1").resolve()
# model = Dia.from_local(
#     config_path="config.json",
#     checkpoint_path=checkpoint_path.as_posix(),
#     compute_dtype="float32",
# )

# text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

# output = model.generate(
#     text,
#     verbose=True,
#     cfg_scale=3.0,
#     temperature=1.8,
#     top_p=0.90,
#     cfg_filter_top_k=50
# )

# model.save_audio("simple2.mp3", output)

checkpoint_path = (Path("weights") / "checkpoint_2000").resolve()
model = Dia.from_local(
    config_path="config.test.json",
    checkpoint_path=checkpoint_path.as_posix(),
    compute_dtype="bfloat16",
)

text = "[S1] nou ja het verwijt was natuurlijk het is te weinig en... [S2] ja."
text_two = "[S1] nee dat uh die recessie ga je natuurlijk niet voorkomen met je met je werk. [S2] nee."
text_three = "[S1] daarop aanhakend wat ik me nog echt kon herinneren was het moment dat ik voor 't eerst m'n bril opzette."
text_four = "[S1] en hoe hoe uhm wat h wat zorgde d'rvoor dat je n stopte? [S2] de Detox-kuur en me me me vrijwillig aanmelden bij verslavingsklinieken omdat ik er omdat ik ervan af wilde. [S1] en waar bedoel..."

output = model.generate(
    text_four,
    verbose=True,
    cfg_scale=0.0,
    # To replicate with non-randomness
    temperature=0.0,
    top_p=1,
    cfg_filter_top_k=None,
    max_tokens=300,
)

model.save_audio("val.mp3", output)
