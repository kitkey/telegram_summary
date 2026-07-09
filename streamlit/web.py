import streamlit as st
import requests


code_input = st.text_input("Введите код")
if st.button("Отправить код"):
    requests.post("http://conn_service:8000/post_code", json=code_input, timeout=30)
textbox_id = st.text_input(label='ID нужного чата')
textbox_subid = st.text_input(label='ID подгруппы чата (опционально)')
textbox_num = st.text_input(label='Количество сообщений для обработки', value="1000")

if st.button(label='Отправить на обработку'):
    params = {
        "group_id": textbox_id,
        "num_messages": int(textbox_num),
    }
    if textbox_subid.strip() != '':
        params["subgroup_id"] = int(textbox_subid.strip())

    url = "http://conn_service:8000/get_subgroup_data"

    response = requests.get(url, params=params, timeout=300).json()

    texts = response.get("messages", [])

    for link, summary in texts:
        if isinstance(summary, list) and summary and isinstance(summary[0], dict):
            summary = summary[0].get('summary_text', '')
        st.write(summary)
        st.link_button("Перейти к сообщению", link)

