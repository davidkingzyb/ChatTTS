import ChatTTS
from IPython.display import Audio
import torchaudio,torch
import soundfile

chat = ChatTTS.Chat()
chat.load_models(compile=False) # Set to True for better performance

texts = ["大多数时候promise是一个可选项，VS Code调用插件之后，它能直接处理正常的返回结果也能处理Thenable的结果类型。当promise是可选的API返回结果时，API会在返回类型中用Thenable表示。",]

wavs = chat.infer(texts, )

soundfile.write("output1.wav", wavs[0][0], 24000)