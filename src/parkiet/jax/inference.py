from parkiet.jax.model import Dia
from pathlib import Path
import jax
import logging

# Enable JAX compilation logging
# jax.config.update("jax_log_compiles", True)
jax_logger = logging.getLogger(__name__)
jax_logger.setLevel(logging.DEBUG)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

checkpoint_path = (Path("weights") / "dia-nl-v1").resolve()
model = Dia.from_local(
    config_path="config.json",
    checkpoint_path=checkpoint_path.as_posix(),
    compute_dtype="float32",
    param_dtype="float32",
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

output = model.generate_jit(
    example_prompts[0],
    # Provide an audio prompt for voice cloning
    # Make sure to prefix the prompt with the spoken audio provided for voice cloning
    # audio_prompt="samples/clone_long.mp3",
    verbose=True,
    cfg_scale=3.0,
    temperature=1.8,
    top_p=0.90,
    cfg_filter_top_k=50,
    # Limit the amount of tokens to prevent lengthy hallucinations when needed
    # 86 tokens = 1 second of audio
    # max_tokens=800,
)

model.save_audio(f"example.mp3", output)
