from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from translation import Translator
import os


class TranslationRequest(BaseModel):
    text: str
    src_lang: str = "eng_Latn" # or "deu_Latn"
    tgt_lang: str = "fer_Latn"
    by_sentence: bool = True
    preprocess: bool = True


app = FastAPI()
translator = Translator()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.post("/translate")
def translate(request: TranslationRequest):
    """
    Perform translation with a fine-tuned NLLB model.
    The language codes are supposed to be in 8-letter format, like "eng_Latn".
    Their list can be returned by /list-languages.
    """
    output = translator.translate(
        request.text,
        src_lang=request.src_lang,
        tgt_lang=request.tgt_lang,
        by_sentence=request.by_sentence,
        preprocess=request.preprocess,
    )
    return {"translation": output}

@app.post("/translate/stream")
async def stream(request: TranslationRequest):
    return StreamingResponse(translator.stream(
        request.text,
        src_lang=request.src_lang,
        tgt_lang=request.tgt_lang,
        by_sentence=request.by_sentence,
        preprocess=request.preprocess,
    ), media_type='text/event-stream')

@app.get("/list-languages")
def list_languages():
    """Show the mapping of supported languages: from their English names to their 8-letter codes."""
    return translator.languages



@app.websocket("/ws/stream/{client_id}")
async def stream_ger_translation(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            print(data)
            text = data["text"]
            lang = data["lang"]
            async for new_text in translator.stream(text, src_lang=lang, tgt_lang="fer_Latn"):
                await websocket.send_text(new_text)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         data = await websocket.receive_text()
#         await websocket.send_text(f"Message text was: {data}")

if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
