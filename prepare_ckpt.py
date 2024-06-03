
from transformers import WhisperProcessor, WhisperForConditionalGeneration 
from funasr import AutoModel

device="cuda:0"
model_id = "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)

model = AutoModel(model="paraformer-zh")
