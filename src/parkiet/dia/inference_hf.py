from transformers import AutoProcessor, DiaForConditionalGeneration


torch_device = "cuda"
model_checkpoint = "pevers/parkiet"

text = [
    "[S1] denk je dat je een open source model kan trainen met weinig geld en middelen? [S2] ja, ik denk het wel. [S1] oh ja, hoe dan? [S2] nou kijk maar in de repo op Git Hub of Hugging Face."
]
processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(text=text, padding=True, return_tensors="pt").to(torch_device)

model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
outputs = model.generate(
    **inputs,
    max_new_tokens=3072,
    guidance_scale=3.0,
    temperature=1.8,
    top_p=0.90,
    top_k=50,
)

outputs = processor.batch_decode(outputs)
processor.save_audio(outputs, "example.mp3")
