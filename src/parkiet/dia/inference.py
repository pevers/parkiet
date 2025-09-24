from parkiet.dia.model import Dia
import torch

torch.manual_seed(42)

model = Dia.from_local(
    config_path="config.json",
    checkpoint_path="weights/dia-nl-v1.pth",
    compute_dtype="bfloat16",
    # device=torch.device("cpu")
)

example_prompts = [
    "[S1] denk je dat je een open source model kan trainen met weinig geld en middelen? [S2] ja, ik denk het wel. [S1] oh ja, hoe dan? [S2] nou kijk maar in de repo op Git Hub of Hugging Face.",
    "[S1] hoeveel stemmen worden er ondersteund? [S2] nou, uhm, ik denk toch wel meer dan twee. [S3] ja, ja, d dat is het mooie aan dit model. [S4] ja klopt, het ondersteund tot vier verschillende stemmen per prompt.",
    "[S1] h h et is dus ook mogelijk, om eh ... uhm, heel veel t te st stotteren in een prompt.",
    "[S1] (laughs) luister, ik heb een mop, wat uhm, drinkt een webdesigner het liefst? [S2] nou ... ? [S1] Earl Grey (laughs). [S2] (laughs) heel goed.",
    # Voice cloning example
    # "[S1] je hebt maar weinig audio nodig om een stem te clonen de rest van deze tekst is uitgesproken door een computer. [S2] wauw, dat klinkt wel erg goed. [S1] ja, ik hoop dat je er wat aan hebt.",
    "[S1] dit is nog een test waarin ik de hele tien seconden probeer vol te praten zodat ik genoeg audio heb zodat ik een stem kan clonen in Eleven Labs dit is wel angstaanjagend dat het zo goed werkt.",
]

output = model.generate(
    example_prompts[3],
    use_torch_compile=False,
    verbose=True,
    cfg_scale=3.0,
    temperature=1.8,
    top_p=0.90,
    cfg_filter_top_k=50,
)
model.save_audio(f"example.mp3", output)
