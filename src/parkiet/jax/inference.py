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
#     cfg_filter_top_k=50,
# )

# model.save_audio("simple.mp3", output)

checkpoint_path = (Path("weights") / "training/checkpoint_1000").resolve()
model = Dia.from_local(
    config_path="config.test.json",
    checkpoint_path=checkpoint_path.as_posix(),
    compute_dtype="float32",
)

text = "[S1] als we kijken naar woonruimte dan is er eigenlijk een fixatie op groter. dus waar we in negentienhonderd gemiddeld tien vierkante meter woonruimte per persoon hadden is dat vandaag de dag drieënvijftig vierkante meter per persoon. [S2] even ter vergelijking. buurlanden als Duitsland en België hebben zevenenveertig vierkante meter per persoon. en in Oost-Europa is 't aantal vierkante meters nog een stuk lager. [S1] en uh we zijn voor die woningen ook meer grond gaan gebruiken."

output = model.generate(
    text,
    verbose=True,
    cfg_scale=0.0,
    temperature=1.8,
    top_p=0.90,
    cfg_filter_top_k=50,
)

model.save_audio("simple.mp3", output)
