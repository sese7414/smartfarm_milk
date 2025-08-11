import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(layout="wide")
st.markdown("""
    <style>
    /* 컨테이너 최대 폭 제거 */
    section.main > div.block-container {
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* 숫자 입력 & 텍스트 입력 */
input[type="number"], input[type="text"] {
    background-color: #E8F6EE !important;
}
/* 숫자 입력 위아래 + / - 버튼 */
div[data-testid="stNumberInput"] button {
    background-color: #E8F6EE !important;
    border: none !important;
}
/* 드롭다운(selectbox) */
div[data-baseweb="select"] > div {
    background-color: #E8F6EE !important;
}
/* 멀티라인 입력창(text_area) */
textarea {
    background-color: #E8F6EE !important;
}
""", unsafe_allow_html=True)

data = joblib.load("regressormodel.pkl")
model = data["model"]
X_columns = data["X_columns"]
cat_features = data["cat_features"]

st.title("🥛 착유량 예측 & 위험 진단")

col1, col2 = st.columns([1, 1])

with col1:
    with st.container(border=True, height=700):
        st.subheader("📥 Feature 입력")
        st.write("")
        # 좌/우 컬럼 나누기
        c0, c1, c2, c3, c4 = st.columns([0.5, 2, 0.5, 2, 0.5])
        with c1:
            milking_count = st.number_input("착유회차", min_value=0, step=1, help="금일의 착유회차")
            cow_age = st.number_input("나이", min_value=0, step=1, help="소의 나이")
            milking_time_spent = st.number_input("착유소요시간", min_value=0, step=1, help="착유 시작부터 끝까지의 시간")
            conductivity = st.number_input("전도도", format="%.2f", help="물질이 전기를 얼마나 잘 전달하는지를 나타내는 물리량")
            temperature = st.number_input("온도", format="%.2f", help="젖소 표면 체온 ")
            fat = st.number_input("유지방", format="%.2f", help="우유에 들어 있는 지방")
            is_individual_issue = st.checkbox("개체문제의심", help="개체에 문제가 의심되는지 여부")
        with c3:
            protein = st.number_input("유단백", format="%.2f", help="우유에 들어있는 단백질")
            air_flow = st.number_input("공기흐름", format="%.2f", help="기계의 공기 압력")
            avg_humidity = st.number_input("평균습도", format="%.2f", help="금일 평균습도")
            avg_temperature = st.number_input("평균기온", format="%.2f", help="금일 평균기온")
            sunshine_duration = st.number_input("일조량", format="%.2f", help="금일 일조량")
            time_of_day = st.selectbox("시간대", options=['야간', '오전', '오후', '저녁'], help="착유를 진행하는 시간대")
            is_machine_issue = st.checkbox("기계결함의심", help="기계 결함이 의심되는지 여부")
with col2:
    with st.container(border=True, height=320):
        st.subheader(
            "예상결과"
        )

        # 좌우 컬럼 나누기
        left_box, right_box = st.columns(2)

        # 왼쪽 컬럼 → 예측 버튼 + 예측 결과
        with left_box:
            st.write(" ")
            clicked = st.button("착유량 예측", key="predict_button", use_container_width=True)

            # 버튼 바로 아래에 결과 자리 확보
            pred_placeholder = st.empty()

            if clicked:
                input_df = pd.DataFrame([{
                    '착유회차' : milking_count,
                    '전도도' : conductivity,
                    '온도' : temperature,
                    '유지방' : fat,
                    '유단백' : protein,
                    '공기흐름' : air_flow,
                    '나이' : cow_age,
                    '평균습도' : avg_humidity,
                    '평균기온' : avg_temperature,
                    '일조량' : sunshine_duration,
                    '착유소요시간' : milking_time_spent,
                    '시간대' : time_of_day,
                    '개체문제의심' : is_individual_issue,
                    '기계결함의심' : is_machine_issue
                }])

                input_df = input_df[X_columns]
                y_pred = model.predict(input_df)[0]
                st.session_state["y_pred"] = y_pred

                # 버튼 바로 아래에 결과 출력
                
                pred_placeholder.markdown(f"#### **예측 착유량:** {y_pred:.2f}L")
#                 pred_placeholder.markdown(  ####### 이거이거이거 가운데 정렬~~~~~~~~~~~~
#     f"<h3 style='text-align:center;'>예측 착유량: {y_pred:.2f}L</h3>",
#     unsafe_allow_html=True
# )
            else:
                # 버튼 바로 아래에 안내 문구
                pred_placeholder.markdown("⬅️ **먼저 왼쪽에 값을 입력해주세요.**")

        # 오른쪽 컬럼 → 버튼 → 결과 문구 → 실제값 입력
        with right_box:
            st.write("")
            clicked_check = st.button("결과 확인", key="check_button", use_container_width=True)

            # 버튼 바로 아래 결과 자리
            error_value = st.empty()

            # 버튼 아래 실제값 입력창
            milk = st.number_input("실제 착유량", format="%.2f", help="실제 착유량을 입력해주세요.")

            if clicked_check:
                if "y_pred" in st.session_state:
                    error = np.abs(milk - st.session_state["y_pred"])
                    st.session_state["error"] = error
                    error_value.markdown(f"**오차(절대값):** {error:.2f}L")
                else:
                    error_value.markdown("⚠️ **먼저 예측을 진행해주세요.**")
            else:
                error_value.markdown("⬅️ **먼저 예측 착유량을 계산해주세요.**")

    with st.container(border=True, height=360):
        image_placeholder = st.empty()

        if "error" in st.session_state:
            error = st.session_state["error"]
            if error <= 2.94:
                status = "정상"
                img_url = "images/green.png"
            elif error <= 4.44:
                status = "주의"
                img_url = "images/yellow.png"
            else:
                status = "위험"
                img_url = "images/red.png"

            with image_placeholder.container():
                _, center, _ = st.columns([1, 2, 1])
                with center:
                    st.image(img_url, caption=f"{status} 상태", width=320) # 
                    # st.markdown(
                    #     f"""
                    #     <div style="text-align: center;">
                    #         <p style="font-size:20px; font-weight:bold;">
                    #             {status} 상태
                    #         </p>
                    #     </div>
                    #     """,
                    #     unsafe_allow_html=True
                    # )
        else:
            image_placeholder.write("결과 확인 버튼을 눌러 이미지를 확인하세요.")