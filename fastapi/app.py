import os
from typing import Dict, Any, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, Body

from telethon import TelegramClient
from telethon.sessions import MemorySession
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GenerationConfig, AutoModelForSeq2SeqLM, pipeline

from chat_processing import create_dialog_branches

SESSION_CONST = {
        'session': MemorySession(),
        'api_id': int(os.getenv('API_ID')),
        'api_hash': os.getenv('API_HASH'),
        'device_model': "CustomDevice",
        'system_version': "4.16.30-vxCUSTOM",
        'app_version': "1.0.0"
}
DEVICE = "cpu"
SUM_MODEL_NAME = 'PavelY/ru-mbart-sum'


client = TelegramClient(**SESSION_CONST)

sim_tokenizer = AutoTokenizer.from_pretrained('Den4ikAI/ruBert-base-qa-ranker')
sim_model = AutoModelForSequenceClassification.from_pretrained('Den4ikAI/ruBert-base-qa-ranker')

model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL_NAME)

sum_model = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    generation_config=gen_cfg,
)

PREFIX = "суммаризуй диалог: "

def summarize(text: str) -> str:
    out = sum_model(PREFIX + text, truncation=True)
    return out[0]["summary_text"]

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await client.connect()
    if not await client.is_user_authorized():
        await client.send_code_request(os.getenv('PHONE_NUMBER'))

@app.post('/post_code')
async def post_code(code: str = Body(...)):
    await client.sign_in(os.getenv('PHONE_NUMBER'), code)

@app.get('/get_subgroup_data')
async def get_messages_from_subgroup(
        group_id: str,
        subgroup_id: Optional[int] = None,
        num_messages: int = 1000
) -> Dict[str, Any]:
    channel = await client.get_entity(group_id)

    messages = pd.DataFrame(
        list(
            reversed(
                [
            {
                "id": m.id,
                "user_id": getattr(m.from_id, "user_id", None),
                "date": m.date.isoformat(),
                "text": m.message,
                "reply_to_msg_id": getattr(m.reply_to, "reply_to_msg_id", None)
            }
            async for m in client.iter_messages(channel, reply_to=subgroup_id, limit=num_messages)
        ]
            )
        )
    )
    message_branches = create_dialog_branches(
        data=messages,
        sim_model=sim_model,
        tokenizer=sim_tokenizer,
        subgroup_id=subgroup_id
    )

    summarized_branches = []
    link_root_msg = f"https://t.me/{group_id}/{subgroup_id}/" if subgroup_id is not None else f"https://t.me/{group_id}/"

    for i, dialog in enumerate(message_branches):
        result = summarize(dialog[1])
        summarized_branches.append((link_root_msg+dialog[0], result))

    return {"response": "ok", "messages": summarized_branches}



if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)