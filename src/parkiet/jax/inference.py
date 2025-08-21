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

checkpoint_path = (Path("weights") / "checkpoint_1000").resolve()
model = Dia.from_local(
    config_path="config.test.json",
    checkpoint_path=checkpoint_path.as_posix(),
    compute_dtype="float32",
    param_dtype="float32",
)

texts = [
    "[S1] nou ja het verwijt was natuurlijk het is te weinig en... [S2] ja.",
    "[S1] nee dat uh die recessie ga je natuurlijk niet voorkomen met je met je werk. [S2] nee.",
    "[S1] daarop aanhakend wat ik me nog echt kon herinneren was het moment dat ik voor 't eerst m'n bril opzette.",
    "[S1] neem ik een douche ik maak mezelf helemaal op alsof ik een date heb. uh of a of alsof ik een belangrijke businessmeeting heb. ja? uh dus ik maak me helemaal op zodat ik uh alsof ik naar een businessmeeting vertrek. en dan pak ik de tram 'k pak mijn laptop mee en ik ga in een Starbucks zitten. zo weet je wel hè met mijn laptop in een Starbucks zitten zoals die echte hipsters doen. ja dat doe ik dus hè. dan ik ben één van die hipsters die daar zit. maar waarom doe ik dat wel?",
    "[S1] als we kijken naar woonruimte dan is er eigenlijk een fixatie op groter. dus waar we in negentienhonderd gemiddeld tien vierkante meter woonruimte per persoon hadden is dat vandaag de dag drieënvijftig vierkante meter per persoon. [S2] even ter vergelijking. buurlanden als Duitsland en België hebben zevenenveertig vierkante meter per persoon. en in Oost-Europa is 't aantal vierkante meters nog een stuk lager. [S1] en uh we zijn voor die woningen ook meer grond gaan gebruiken.",
    "[S1] ja en ook wanneer die regeling er dan komt weet Von Hoff nog niet. heeft dus onder andere te maken met Europese staatssteunreugel-regels waar één en ander moet worden getoetst. volgens hem wordt er nu keihard aan gewerkt op het ministerie van Economische Zaken. en verder maakt de MKB-voorman zich uh ja toch wel zorgen om de gevolgen van deze prinsesdag voor zijn achterman.",
    "[S1] ja nou vandaag dus uh uh Prinsjesdag. we zullen heel veel horen en zien natuurlijk vandaag. uh waar waar moeten we allemaal op gaan letten Sofie?",
    "[S1] en hoe hoe uhm wat h wat zorgde d'rvoor dat je n stopte? [S2] de Detox-kuur en me me me vrijwillig aanmelden bij verslavingsklinieken omdat ik er omdat ik ervan af wilde. [S1] en waar bedoel...",
    "[S1] in scenario Groenland is eigenlijk de extra ruimte die we nodig hebben voor wonen eigenlijk 't kleinst van alle vier de scenario's. [S2] scenario drie",
    "[S1] ja hele goeie vraag want he het is blijkbaar zo wijd verspreid dat wel heel veel mensen dat idee op een bepaalde manier hebben. [S2] in ieder geval Jaap en ik. [S1] (laughs). [S2] (laughs). [S1] nou d dat zijn er genoeg. uh ik vroeg aan Denise inderdaad waar dat dan vandaan komt en waarom mensen dat idee soms hebben.",
]

for i, text in enumerate(texts):
    output = model.generate(
        text,
        verbose=True,
        cfg_scale=0.0,
        # To replicate with non-randomness
        temperature=0.0,
        cfg_filter_top_k=1,
        # top_p=0.90,
        max_tokens=300,
    )
    print(f"Writing {text}.mp3")
    model.save_audio(f"{i}.mp3", output)
