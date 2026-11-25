import streamlit as st
from google import genai
from PIL import Image
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. Настройка страницы (боковая панель полностью скрыта)
st.set_page_config(
    page_title="Эмпатика — Платформа заботы о питомцах",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Кастомный CSS для современного дизайна и скрытия технических кнопок Streamlit
st.markdown("""
<style>
    /* Скрываем верхнюю панель с кнопкой Deploy и футер */
    [data-testid="stToolbar"] {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    
    /* Уменьшаем лишний отступ сверху */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }

    /* Красивый главный баннер */
    .hero-card {
        background: linear-gradient(135deg, #2A7B6F 0%, #1B4D46 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px rgba(42, 123, 111, 0.15);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-bottom: 0.8rem;
        letter-spacing: 0.3px;
    }
    .hero-card h1 {
        color: white !important;
        margin: 0 0 0.5rem 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .hero-card p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0;
        font-size: 1.05rem;
        line-height: 1.5;
    }

    /* Стилизация кнопок */
    .stButton > button[kind="primary"] {
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 12px rgba(42, 123, 111, 0.2);
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(42, 123, 111, 0.3);
    }

    /* Стилизация вкладок */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #FFFFFF;
        padding: 8px;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EBF5F3 !important;
        color: #2A7B6F !important;
    }

    /* Карточка пустого состояния */
    .empty-state {
        text-align: center;
        padding: 3rem 1.5rem;
        color: #64748B;
    }
    .empty-state-icon {
        font-size: 2.8rem;
        margin-bottom: 0.8rem;
    }
    
    /* Карточка рекомендации корма */
    .food-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2A7B6F;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
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

# 3. Автоматическое подключение API-ключа из скрытых настроек
def get_client():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        return genai.Client(api_key=api_key)
    except Exception:
        st.error("Не найден API-ключ в настройках сервера (secrets.toml). Обратитесь к администратору проекта.")
        return None

# 4. База кормов для ML-модуля
@st.cache_data
def load_food_data():
    data = [
        {"Название": "Royal Canin Sensitive Skin", "Вид": "Собака", "Возраст_код": 1, "Гипоаллергенный": 1, "Цена_тг": 18500, "Описание": "Специализированный корм для собак с чувствительной кожей и склонностью к аллергии."},
        {"Название": "Purina ONE Adult", "Вид": "Собака", "Возраст_код": 1, "Гипоаллергенный": 0, "Цена_тг": 7500, "Описание": "Сбалансированный повседневный рацион для поддержания иммунитета и активности."},
        {"Название": "Brit Care Puppy Lamb & Rice", "Вид": "Собака", "Возраст_код": 0, "Гипоаллергенный": 1, "Цена_тг": 14000, "Описание": "Беззерновой рацион с ягнёнком и рисом для здорового роста щенков."},
        {"Название": "Chappi Мясное изобилие", "Вид": "Собака", "Возраст_код": 1, "Гипоаллергенный": 0, "Цена_тг": 4200, "Описание": "Доступный базовый вариант для крупных приютов и вольерного содержания."},
        {"Название": "Monge Cat Hypoallergenic", "Вид": "Кошка", "Возраст_код": 1, "Гипоаллергенный": 1, "Цена_тг": 16000, "Описание": "Диетический ветеринарный корм при пищевой непереносимости и дерматитах."},
        {"Название": "Pro Plan Kitten", "Вид": "Кошка", "Возраст_код": 0, "Гипоаллергенный": 0, "Цена_тг": 11500, "Описание": "Рацион для котят до 1 года, укрепляет иммунную систему и пищеварение."},
        {"Название": "Whiskas Вкусные подушечки", "Вид": "Кошка", "Возраст_код": 1, "Гипоаллергенный": 0, "Цена_тг": 3800, "Описание": "Бюджетный повседневный корм для взрослых стерилизованных и домашних кошек."},
        {"Название": "Grandorf Senior Holistic", "Вид": "Кошка", "Возраст_код": 2, "Гипоаллергенный": 1, "Цена_тг": 19000, "Описание": "Холистик-корм для пожилых кошек (7+) с защитой суставов и почек."},
    ]
    return pd.DataFrame(data)

# 5. Главный баннер приложения
st.markdown("""
<div class="hero-card">
    <div class="hero-badge">AI-платформа • Компьютерное зрение + Machine Learning</div>
    <h1>🐾 Эмпатика</h1>
    <p>Интеллектуальный помощник для волонтёров приютов и владельцев питомцев: первичный осмотр по фото, создание анкет для пристройства и умный подбор рациона.</p>
</div>
""", unsafe_allow_html=True)

# 6. Навигация по трём сценариям
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
            health_img_file = st.file_uploader(
                "Загрузите фото проблемной зоны или питомца",
                type=["jpg", "jpeg", "png"],
                key="health_img"
            )
            if health_img_file:
                st.image(health_img_file, caption="Загруженный снимок", use_container_width=True)
                
            symptoms_text = st.text_area(
                "Опишите поведение и симптомы",
                placeholder="Например: кот второй день чешет ухо, появилось небольшое покраснение, аппетит в норме...",
                height=110
            )
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
                                "Ты — ветеринарный ИИ-ассистент для волонтёров приютов и владельцев животных. "
                                "Проанализируй приложенное фото и описание. Структурируй ответ строго по пунктам с красивым форматированием:\n"
                                "1) **Визуальные признаки**: что видно на фото (покраснения, высыпания, состояние шерсти/глаз).\n"
                                "2) **Уровень срочности**: выбери один из статусов — 🟢 Наблюдение дома, 🟡 Плановый визит к ветеринару или 🔴 Срочно в клинику — и объясни почему.\n"
                                "3) **Первичные рекомендации**: безопасные действия по уходу до осмотра врача.\n"
                                "В конце добавь короткий дисклеймер о том, что ответ ИИ не заменяет очный приём ветеринара.\n"
                                f"Описание от пользователя: {symptoms_text}"
                            )
                            contents = [prompt]
                            if health_img_file:
                                contents.append(Image.open(health_img_file))
                            
                            response = client.models.generate_content(model="gemini-2.5-flash", contents=contents)
                            st.markdown(response.text)
                        except Exception as e:
                            st.error(f"Ошибка при обращении к ИИ: {e}")
                elif client:
                    st.warning("Пожалуйста, загрузите фотографию или опишите симптомы слева.")
            else:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-icon">🩺</div>
                    <p><b>Здесь появится разбор состояния питомца</b><br>
                    Загрузите фотографию или опишите симптомы слева, чтобы получить оценку срочности и советы по уходу.</p>
                </div>
                """, unsafe_allow_html=True)

# --- МОДУЛЬ 2: ГЕНЕРАТОР ПОСТОВ ---
with tab2:
    col1, col2 = st.columns([1, 1.15], gap="large")
    
    with col1:
        with st.container(border=True):
            st.subheader("1. Карточка питомца")
            adopt_img_file = st.file_uploader(
                "Фото питомца для публикации",
                type=["jpg", "jpeg", "png"],
                key="adopt_img"
            )
            if adopt_img_file:
                st.image(adopt_img_file, caption="Фото для поста", use_container_width=True)
                
            pet_name = st.text_input("Кличка и возраст", placeholder="Например: Барсик, 8 месяцев")
            pet_traits = st.text_area(
                "Характер, привычки и история",
                placeholder="Ласковый, приучен к лотку, дружит с другими котами, обработан от паразитов и привит.",
                height=110
            )
            post_tone = st.selectbox(
                "Стиль текста",
                ["Трогательный и душевный", "Бодрый и позитивный", "Короткий для сторис и объявлений"]
            )
            gen_post_btn = st.button("✨ Сгенерировать пост для соцсетей", type="primary", use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("2. Готовый текст для Instagram")
            if gen_post_btn:
                client = get_client()
                if client and pet_traits:
                    with st.spinner("Создаём цепляющую анкету для пристройства..."):
                        try:
                            prompt = (
                                f"Напиши готовый продающий пост для Instagram, чтобы найти дом питомцу из приюта. "
                                f"Кличка и возраст: {pet_name}. Характер и детали: {pet_traits}. "
                                f"Тон повествования: {post_tone}. "
                                "Если приложено фото, органично вплети в текст милые детали внешности питомца с фотографии. "
                                "Структура: цепляющий заголовок с эмодзи, короткая история, список плюсов питомца, призыв к действию и 5–7 хэштегов."
                            )
                            contents = [prompt]
                            if adopt_img_file:
                                contents.append(Image.open(adopt_img_file))
                            
                            response = client.models.generate_content(model="gemini-2.5-flash", contents=contents)
                            st.markdown(response.text)
                            st.divider()
                            st.download_button(
                                label="📥 Скачать текст поста (.txt)",
                                data=response.text,
                                file_name=f"post_{pet_name or 'pet'}.txt",
                                mime="text/plain"
                            )
                        except Exception as e:
                            st.error(f"Ошибка при генерации поста: {e}")
                elif client:
                    st.warning("Заполните хотя бы описание характера питомца слева.")
            else:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-icon">📱</div>
                    <p><b>Здесь появится готовый пост для пристройства</b><br>
                    Добавьте фото и пару слов о характере животного — ИИ оформит текст с заголовком и хэштегами.</p>
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
            st.write("")
            hypo = st.checkbox("Гипоаллергенный состав")

        calc_btn = st.button("🥣 Подобрать оптимальный рацион", type="primary")

    if calc_btn:
        subset = df[df["Вид"] == species].reset_index(drop=True)
        
        # ML-модель ближайших соседей (KNN) из scikit-learn
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