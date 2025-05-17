import streamlit as st
import requests


code_input = st.text_input("Введите код")
if st.button("Отправить код"):
    send_code = requests.post("http://conn_service:8000/post_code", data={"code": code_input})
textbox_id = st.text_input(label='ID нужного чата')
textbox_subid = st.text_input(label='ID подгруппы чата (опционально)')
textbox_num = st.text_input(label='Количество сообщений для обработки', value="1000")

if st.button(label='Отправить на обработку'):
    params = {
        "group_id": textbox_id,
        "num_messages": int(textbox_num),
        "subgroup_id": int(textbox_subid.strip()) if textbox_subid.strip() != '' else None
    }

    url = "http://conn_service:8000/get_subgroup_data"

    response = requests.get(url, params=params).json()

    texts = response.get("messages", [])

    for link, summary in texts:
        st.write(summary[0]['summary_text'])
        st.link_button("Перейти к сообщению", link)

