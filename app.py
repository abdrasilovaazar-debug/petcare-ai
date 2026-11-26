import streamlit as st
from google import genai
from PIL import Image
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Эмпатика — Платформа заботы о питомцах",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    [data-testid="stToolbar"] {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    .block-container {
        padding-top: 3.2rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }
    .hero-card {
        background: linear-gradient(135deg, #2A7B6F 0%, #1B4D46 100%);
        padding: 2rem 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px rgba(42, 123, 111, 0.15);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        margin-bottom: 0.7rem;
    }
    .hero-card h1 {
        color: white !important;
        margin: 0 0 0.5rem 0;
        font-size: 2.1rem;
    }
    .hero-card p {
        color: rgba(255, 255, 255, 0.92);
        margin: 0;
        font-size: 1rem;
        line-height: 1.5;
    }
    .stButton > button[kind="primary"] {
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        border: none;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #FFFFFF;
        padding: 8px;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EBF5F3 !important;
        color: #2A7B6F !important;
    }
    .empty-state {
        text-align: center;
        padding: 2.5rem 1rem;
        color: #64748B;
    }
    .empty-state-icon {
        font-size: 2.5rem;
        margin-bottom: 0.6rem;
    }
    .food-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2A7B6F;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
    }
    .food-tag {
        display: inline-block;
        background: #EBF5F3;
        color: #2A7B6F;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 12px;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

def get_client():
    try:
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"].strip())
    except Exception:
        st.error("API-ключ не найден в настройках Secrets.")
        return None

def prepare_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img.thumbnail((1024, 1024))
    return img

def ask_gemini(client, contents):
    models_to_try = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
    errors = []
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(model=model_name, contents=contents)
            return response.text
        except Exception as e:
            errors.append(f"{model_name}: {e}")
    raise RuntimeError("\n".join(errors))

@st.cache_data
def load_food_data():
    data = [
        {"Название": "Royal Canin Sensitive Skin", "Вид": "Собака", "Возраст_код": 1, "Гипоаллергенный": 1, "Цена_тг": 18500, "Описание": "Для собак с чувствительной кожей и склонностью к аллергии."},
        {"Название": "Purina ONE Adult", "Вид": "Собака", "Возраст_код": 1, "Гипоаллергенный": 0, "Цена_тг": 7500, "Описание": "Сбалансированный повседневный рацион для активности и иммунитета."},
        {"Название": "Brit Care Puppy Lamb & Rice", "Вид": "Собака", "Возраст_код": 0, "Гипоаллергенный": 1, "Цена_тг": 14000, "Описание": "Беззерновой рацион с ягнёнком и рисом для роста щенков."},
        {"Название": "Chappi Мясное изобилие", "Вид": "Собака", "Возраст_код": 1, "Гипоаллергенный": 0, "Цена_тг": 4200, "Описание": "Доступный базовый корм для крупных приютов."},
        {"Название": "Monge Cat Hypoallergenic", "Вид": "Кошка", "Возраст_код": 1, "Гипоаллергенный": 1, "Цена_тг": 16000, "Описание": "Диетический корм при пищевой непереносимости и дерматитах."},
        {"Название": "Pro Plan Kitten", "Вид": "Кошка", "Возраст_код": 0, "Гипоаллергенный": 0, "Цена_тг": 11500, "Описание": "Рацион для котят до 1 года, укрепляет иммунитет."},
        {"Название": "Whiskas Вкусные подушечки", "Вид": "Кошка", "Возраст_код": 1, "Гипоаллергенный": 0, "Цена_тг": 3800, "Описание": "Бюджетный рацион для взрослых домашних и приютских кошек."},
        {"Название": "Grandorf Senior Holistic", "Вид": "Кошка", "Возраст_код": 2, "Гипоаллергенный": 1, "Цена_тг": 19000, "Описание": "Холистик-корм для пожилых кошек (7+) с защитой почек и суставов."},
    ]
    return pd.DataFrame(data)

st.markdown("""
<div class="hero-card">
    <div class="hero-badge">AI-платформа • Компьютерное зрение + Machine Learning</div>
    <h1>🐾 Эмпатика</h1>
    <p>Интеллектуальный помощник для волонтёров приютов и владельцев питомцев: первичный осмотр по фото, создание анкет для пристройства и умный подбор рациона.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🩺 Анализ симптомов по фото",
    "📱 Генератор постов для пристройства",
    "🥣 ML-подбор рациона"
])

# --- МОДУЛЬ 1: АНАЛИЗ СИМПТОМОВ ---
with tab1:
    col1, col2 = st.columns([1, 1.15], gap="large")
    with col1:
        with st.container(border=True):
            st.subheader("1. Исходные данные")
            health_img_file = st.file_uploader("Загрузите фото проблемной зоны или питомца", type=["jpg", "jpeg", "png"], key="health_img")
            if health_img_file:
                st.image(health_img_file, caption="Загруженный снимок", use_container_width=True)
            symptoms_text = st.text_area("Опишите поведение и симптомы", placeholder="Например: кот второй день чешет ухо, появилось покраснение...", height=110)
            analyze_btn = st.button("🔍 Проанализировать состояние", type="primary", use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("2. Заключение ИИ-ассистента")
            if analyze_btn:
                client = get_client()
                if client and (health_img_file or symptoms_text):
                    with st.spinner("Нейросеть изучает изображение и симптомы..."):
                        try:
                            prompt = (
                                "Ты — ветеринарный ИИ-ассистент для приютов и владельцев животных. "
                                "Проанализируй фото и описание по пунктам:\n"
                                "1) **Визуальные признаки**: что видно на фото.\n"
                                "2) **Уровень срочности**: 🟢 Наблюдение дома, 🟡 Плановый визит к врачу или 🔴 Срочно в клинику.\n"
                                "3) **Первичные рекомендации**: безопасный уход до осмотра.\n"
                                "Добавь короткий дисклеймер, что это не заменяет приём ветеринара.\n"
                                f"Описание: {symptoms_text}"
                            )
                            contents = [prompt]
                            if health_img_file:
                                contents.append(prepare_image(health_img_file))
                            result = ask_gemini(client, contents)
                            st.markdown(result)
                        except Exception as e:
                            st.error(f"Ошибка при обращении к ИИ:\n{e}")
                elif client:
                    st.warning("Загрузите фотографию или опишите симптомы слева.")
            else:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-icon">🩺</div>
                    <p><b>Здесь появится разбор состояния питомца</b><br>Загрузите фото или опишите симптомы слева.</p>
                </div>
                """, unsafe_allow_html=True)

# --- МОДУЛЬ 2: ГЕНЕРАТОР ПОСТОВ ---
with tab2:
    col1, col2 = st.columns([1, 1.15], gap="large")
    with col1:
        with st.container(border=True):
            st.subheader("1. Карточка питомца")
            adopt_img_file = st.file_uploader("Фото питомца для публикации", type=["jpg", "jpeg", "png"], key="adopt_img")
            if adopt_img_file:
                st.image(adopt_img_file, caption="Фото для поста", use_container_width=True)
            pet_name = st.text_input("Кличка и возраст", placeholder="Например: Барсик, 8 месяцев")
            pet_traits = st.text_area("Характер, привычки и история", placeholder="Ласковый, приучен к лотку, привит.", height=110)
            post_tone = st.selectbox("Стиль текста", ["Трогательный и душевный", "Бодрый и позитивный", "Короткий для сторис и объявлений"])
            gen_post_btn = st.button("✨ Сгенерировать пост для соцсетей", type="primary", use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("2. Готовый текст для Instagram")
            if gen_post_btn:
                client = get_client()
                if client and pet_traits:
                    with st.spinner("Создаём текст для пристройства..."):
                        try:
                            prompt = (
                                f"Напиши продающий пост для Instagram, чтобы найти дом питомцу из приюта. "
                                f"Кличка и возраст: {pet_name}. Характер: {pet_traits}. Тон: {post_tone}. "
                                "Если есть фото, опиши милые детали внешности. Добавь заголовок, плюсы питомца, призыв и 5 хэштегов."
                            )
                            contents = [prompt]
                            if adopt_img_file:
                                contents.append(prepare_image(adopt_img_file))
                            result = ask_gemini(client, contents)
                            st.markdown(result)
                            st.divider()
                            st.download_button("📥 Скачать текст поста (.txt)", data=result, file_name=f"post_{pet_name or 'pet'}.txt", mime="text/plain")
                        except Exception as e:
                            st.error(f"Ошибка при генерации поста:\n{e}")
                elif client:
                    st.warning("Заполните описание характера питомца слева.")
            else:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-icon">📱</div>
                    <p><b>Здесь появится готовый пост для пристройства</b><br>Добавьте фото и пару слов о характере животного.</p>
                </div>
                """, unsafe_allow_html=True)

# --- МОДУЛЬ 3: ML-ПОДБОР РАЦИОНА ---
with tab3:
    with st.container(border=True):
        st.subheader("Параметры питомца и бюджет")
        df = load_food_data()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            species = st.selectbox("Вид животного", ["Кошка", "Собака"])
        with c2:
            age_label = st.selectbox("Возрастная группа", ["Малыш (до 1 года)", "Взрослый (1–7 лет)", "Пожилой (7+ лет)"])
            age_map = {"Малыш (до 1 года)": 0, "Взрослый (1–7 лет)": 1, "Пожилой (7+ лет)": 2}
        with c3:
            budget = st.slider("Бюджет (тг / уп.)", min_value=3000, max_value=22000, value=12000, step=500)
        with c4:
            st.write("")
            hypo = st.checkbox("Гипоаллергенный состав")
        calc_btn = st.button("🥣 Подобрать оптимальный рацион", type="primary")

    if calc_btn:
        subset = df[df["Вид"] == species].reset_index(drop=True)
        features = ["Возраст_код", "Гипоаллергенный", "Цена_тг"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(subset[features])
        user_vector = scaler.transform([[age_map[age_label], int(hypo), budget]])
        knn = NearestNeighbors(n_neighbors=min(2, len(subset)), metric="euclidean")
        knn.fit(X_scaled)
        distances, indices = knn.kneighbors(user_vector)

        st.markdown("### Рекомендованные рационы (алгоритм K-Nearest Neighbors):")
        for dist, idx in zip(distances[0], indices[0]):
            row = subset.iloc[idx]
            match_score = max(65, min(99, int(100 - dist * 12)))
            hypo_tag = "Гипоаллергенный" if row["Гипоаллергенный"] == 1 else "Стандартный состав"
            st.markdown(f"""
            <div class="food-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 1.15rem; font-weight: 700; color: #1E293B;">{row['Название']}</span>
                    <span style="font-size: 1.1rem; font-weight: 700; color: #2A7B6F;">{row['Цена_тг']:,} ₸</span>
                </div>
                <div style="margin-bottom: 8px;">
                    <span class="food-tag">Точность совпадения: {match_score}%</span>
                    <span class="food-tag">{hypo_tag}</span>
                </div>
                <p style="margin: 0; color: #475569; font-size: 0.95rem;">{row['Описание']}</p>
            </div>
            """, unsafe_allow_html=True)
