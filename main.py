from dotenv import load_dotenv
load_dotenv()  # must be first, before anything uses env vars

import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import google.genai as genai
from google.genai import types

app = FastAPI()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

@app.get("/tts")
def tts(prompt: str):
    return StreamingResponse(
        stream_tts(llmPipeline(prompt)),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": "24000",
            "X-Bit-Depth": "16",
            "X-Channels": "1"
        }
    )


def stream_tts(text):
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    for sentence in sentences:
        print(f"TTS: {sentence}")
        try:
            for chunk in client.models.generate_content_stream(
                model="gemini-2.5-flash-preview-tts",
                contents=sentence,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Sadachbia")
                        )
                    ),
                )
            ):
                if not chunk.candidates:
                    continue
                if not chunk.candidates[0].content:
                    continue
                if not chunk.candidates[0].content.parts:
                    continue
                yield chunk.candidates[0].content.parts[0].inline_data.data
        except Exception as e:
            print(f"TTS ERROR: {e}")
            raise

def askLLM(prompt, meta="", context="", model="gemini-2.5-flash"):
    response = client.models.generate_content(
        model=model, contents=f'''
    {meta}
    {prompt}
    {context}
    '''
    )
    return response.text

def llmPipeline(prompt):
    context = getMCPContext(prompt)
    if context:
        context = "use the context: " + context
    meta = "this response will be spoken aloud, use plain language with no markdown or special characters: "
    return askLLM(prompt, meta, context)


def getWeather():
    return "Worcester: 67 degrees and sunny with a chance of rain at noon"

functions = [
    {"name": "getWeather", "description": "receives weather in Worcester", "function": getWeather},
]

def getMCPContext(prompt):
    function_list = "\n".join([f"name: '{f['name']}', description: '{f['description']}'" for f in functions])
    query = f"""
    Would this query need context from any of the following functions?
    Query: {prompt}
    
    Functions:
    {function_list}
    
    Reply with ONLY the function name exactly as written, or the single word 'none'. No punctuation, no explanation.
    """
    fun = askLLM(query, model="gemini-2.5-flash-lite").strip().strip("'\"").strip()
    print(f"MCP selected: '{fun}'")
    
    if fun.lower() == 'none':
        return ''
    
    for f in functions:
        if f['name'].lower() == fun.lower():
            return f['function']()
    
    return ''

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)