import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import us
import joblib
from matplotlib.ticker import FuncFormatter
from huggingface_hub import hf_hub_download
st.set_page_config(initial_sidebar_state="expanded")
st.markdown("""
    <style>
    /* --- General & Input Styles (No Change) --- */
    .main { padding: 20px; }
    .title { text-align: center; font-size: 2em; color: #4CAF50; }
    div[data-baseweb="select"] > div:first-child,
    div.stTextInput>div[data-baseweb="input"],
    div.stDateInput>div[data-baseweb="input"],
    div.stTimeInput>div[data-baseweb="input"],
    .stNumberInput input[type="number"] {
        border: 1px solid #A9A9A9 !important;
        background-color: #FFFFFF !important;
        border-radius: 0.3rem !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        border: 2px solid #000000;
        background-color: #FFFFFF;
        color: #000000;
        border-radius: 0.5rem;
        padding: 10px 24px;
        font-weight: bold;
        font-size: 1.1em;
    }
    [data-testid="stSidebar"] .stButton > button:active {
        background-color: #E0E0E0;
        border-color: #000000;
    }

    /* PART 1: Styles the 'CLOSE' button when sidebar is open */
    [data-testid="stSidebarCollapseButton"] {
        /* Positioning */
        position: fixed !important;
        left: 15px !important;
        top: 15px !important;
        z-index: 1000 !important;
        width: 45px !important;
        height: 45px !important;
        
        /* Appearance */
        background-color: #FFFFFF !important;
        border: 2px solid #000000 !important;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2) !important;
        border-radius: 10px !important;

        /* --- NEW BRUTE FORCE VISIBILITY RULES --- */
        display: block !important;
        opacity: 1 !important;
        visibility: visible !important;
        transform: none !important;
        -webkit-transform: none !important;
    }
    [data-testid="stSidebarCollapseButton"] svg {
        display: none !important;
    }
    [data-testid="stSidebarCollapseButton"]::after {
        content: '☰' !important;
        color: #4CAF50 !important;
        font-size: 28px !important;
        font-weight: bold !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        pointer-events: none !important;
    }

    /* PART 2: Styles the 'OPEN' button (No Change to this part) */
    button[data-testid="stBaseButton-headerNoPadding"] {
        position: fixed !important;
        left: 15px !important;
        top: 15px !important;
        z-index: 1000 !important;
        width: 45px !important;
        height: 45px !important;
        background-color: #FFFFFF !important;
        border: 2px solid #000000 !important;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2) !important;
        border-radius: 10px !important;
    }
    button[data-testid="stBaseButton-headerNoPadding"] svg {
        display: none !important;
    }
    button[data-testid="stBaseButton-headerNoPadding"]::after {
        content: '☰' !important;
        color: #4CAF50 !important;
        font-size: 28px !important;
        font-weight: bold !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        transform: translate(-50%, -50%) !important;
        pointer-events: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
repo_id = "tomasssrm/house-price-prediction-app"
@st.cache_data
def load_clean_data():
    data_file_path = hf_hub_download(repo_id=repo_id, filename="Real_Estate_Model.csv")
    df = pd.read_csv(data_file_path)
    df['city'] = df['city'].astype('category')
    df['state'] = df['state'].astype('category')
    df['zip_code'] = df['zip_code'].astype('category')
    df['bed'] = df['bed'].astype(int)
    df['bath'] = df['bath'].astype(int)
    return df

@st.cache_resource
def load_model():
    model_file_path = hf_hub_download(repo_id=repo_id, filename="tuned_lgbm_model.pkl")
    model = joblib.load(model_file_path)
    return model

@st.cache_resource
def load_low_model():
    low_model_file_path = hf_hub_download(repo_id=repo_id, filename="tuned_lgbm_model_below_300k.pkl")
    model_low = joblib.load(low_model_file_path)
    return model_low

data = load_clean_data()
model = load_model()
model_low = load_low_model()

#App layout
st.sidebar.title("Navigation")
page = st.sidebar.radio("**Select a page**", ["Prediction", "Market Visualizations"])
st.sidebar.title("Enter House Details")

#Initialize session state variables
if 'state' not in st.session_state:
    st.session_state['state'] = data['state'].unique()[1]
if 'city' not in st.session_state:
    st.session_state['city'] = None

#State selection
state_options = data['state'].unique()
state = st.sidebar.selectbox("**State**", options=state_options, index=list(state_options).index(st.session_state['state']))
st.session_state['state'] = state

# Filter cities based on selected state
filtered_cities = data[data['state'] == state]['city'].unique()

# Reset city selection if it's no longer valid
if st.session_state['city'] not in filtered_cities:
    st.session_state['city'] = filtered_cities[0]

# City selection
city = st.sidebar.selectbox("**City**", options=filtered_cities, index=list(filtered_cities).index(st.session_state['city']))
st.session_state['city'] = city

# Filter zip codes based on selected city
filtered_zip_codes = data[(data['state'] == state) & (data['city'] == city)]['zip_code'].unique()

# Other inputs
bed = st.sidebar.number_input("**Number of Bedrooms**", min_value=1, max_value=10, value=3, step=1)
bath = st.sidebar.number_input("**Number of Bathrooms**", min_value=1, max_value=10, value=3, step=1)
house_size = st.sidebar.number_input("**House Size (SQFT)**", min_value=200, max_value=10000, value=2000, step=500)
predict_button = st.sidebar.button("**Predict House Price**")

if page == "Prediction":
    st.markdown("<div class='title'>House Price Predictor</div>", unsafe_allow_html=True)
    st.write("""
    **Welcome!** This app predicts house prices based on key features like the number of bathrooms, bedrooms, house size, and location (state and city). Adjust the inputs in the sidebar to see the estimated price and potential down payment. Explore the 'Market Visualizations' page (select from the sidebar at the left of the page) for more in-depth charts and housing market trends.
    """)

    if predict_button:
        feature_columns = ['bed', 'bath', 'state', 'city', 'house_size']
        input_data = pd.DataFrame({
            "state": [state],
            "city": [city],
            "bed": [bed],
            "bath": [bath],
            "house_size": [house_size]
        })
        input_data['state'] = input_data['state'].astype('category')
        input_data['city'] = input_data['city'].astype('category')
        input_data = input_data[feature_columns] 

        initial_prediction = model.predict(input_data)[0]
        if initial_prediction < 300000:
            prediction = model_low.predict(input_data)[0]
        else:
            prediction = initial_prediction

        rounded_prediction = round(prediction)
        st.success(f"The predicted price for the house is **${rounded_prediction:,.2f}**.")

        # Down Payment Estimates
        st.write("### Down Payment Estimates")
        fha_down_payment = 0.035 * rounded_prediction
        conventional_loan = 0.2 * rounded_prediction
        st.write(f"**FHA Loan Down Payment (3.5%)**: ${fha_down_payment:,.2f}")
        st.write(f"**Conventional Loan Down Payment (20%)**: ${conventional_loan:,.2f}")

        # Visualizations
        st.write(f"### House Price Distribution in {state}")
        state_data = data[data['state'] == state]
        median_price = state_data['price'].median()
        fig, ax = plt.subplots(figsize=(10,6))

        # Histogram with different KDE line color
        sns.histplot(state_data['price'], kde=True, bins=30, color="skyblue", ax=ax)
        sns.kdeplot(state_data['price'], color="darkblue", linewidth=2, ax=ax)  # KDE in dark blue

        # Median line in red with label
        ax.axvline(median_price, color='red', linestyle='dashed', linewidth=2, label=f"Median: ${median_price:,.0f}")

        # Labels and legend
        ax.set_title(f"House Price Distribution in {state}", fontsize=14)
        ax.set_xlabel("Price ($)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.legend()

        st.pyplot(fig)

#Page 2
if page == "Market Visualizations":
    st.markdown("<div class='title'>Housing Market Visualizations</div>", unsafe_allow_html=True)
    st.write("Explore interactive charts and graphs to discover information, trends, and relationships within the housing market data.")

    # 1. Distribution of House Prices
    st.subheader("Distribution of House Prices by State")
    @st.cache_data
    def get_filtered_data(state):
        return data[data['state'] == state]
    selected_state = st.selectbox("Select the State of Interest", options=data['state'].unique())
    filtered_state_data = get_filtered_data(selected_state)
    median_price2 = filtered_state_data['price'].median()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.histplot(filtered_state_data['price'], kde=True, ax=ax, color="skyblue")
    sns.kdeplot(filtered_state_data['price'], color="darkblue", linewidth=2, ax=ax)
    ax.axvline(median_price2, color='red', linestyle='dashed', linewidth=2, label=f"Median: ${median_price2:,.0f}")
    ax.set_title(f"Distribution of House Prices in {selected_state}")
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/1000):,}k'))
    ax.legend()
    st.pyplot(fig)

    #2. Map of state per sqft
    st.subheader("Median Price per Square Feet by Price")
    state_abbrev_map = {state.name: state.abbr for state in us.states.STATES}
    data_map = data.copy()
    data_map['state'] = data_map['state'].map(state_abbrev_map)
    state_sqft_price = (
        data_map.groupby("state")["Price_per_sqft"].median().reset_index())
    state_sqft_price = state_sqft_price.rename(columns={"Price_per_sqft": "median_price_per_sqft"})
    fig = px.choropleth(
    state_sqft_price,
    locations="state", 
    locationmode="USA-states",  
    color="median_price_per_sqft", 
    color_continuous_scale="Viridis", 
    scope="usa",  
    title="Median Price per Square Foot by State",
    labels={"median_price_per_sqft": "Price per Sqft ($)"}
    )
    st.plotly_chart(fig)

    #3. Top 10 cities with highest median price
    st.subheader("Top 10 Cities with Highest Median Price")
    state_list_top_10 = data['state'].unique()
    selected_state_top_10 = st.selectbox("Select the State of Interest", options=sorted(state_list_top_10))
    filtered_state_data_top_10 = data[data['state'] == selected_state_top_10]
    top_10_cities = (
        filtered_state_data_top_10.groupby('city')['price']
        .median()
        .sort_values(ascending=False)
        .head(10)
        .reset_index())
    fig = px.bar(
        top_10_cities,
        x="city",
        y="price",
        title=f"Top 10 Cities in {selected_state_top_10} by Median Price",
        labels={"price": "Median Price ($)", "city": "City"},
        color="price",
        color_continuous_scale="Blues"
    )
    fig.update_layout(
    xaxis_tickfont_color='black', 
    yaxis_tickfont_color='black',  
    xaxis_title_font_color='black',
    yaxis_title_font_color='black',
    legend_font_color='black',
    coloraxis_colorbar_tickfont_color='black', 
    coloraxis_colorbar_title_font_color='black'
    )
    st.plotly_chart(fig)

    #4. Bottom 10 Cities by median House Price
    st.subheader("Bottom 10 Cities by Median House Price")
    selected_state_bottom_10 = st.selectbox("Select the State of Interest", options=data['state'].unique(), key="Bottom_10_state_selectbox")
    filtered_state_data_bottom_10 = data[data['state'] == selected_state_bottom_10]
    bottom_10_cities = (
        filtered_state_data_bottom_10.groupby('city')['price']
        .median()
        .sort_values(ascending=True)
        .head(10)
        .reset_index()
    )
    fig = px.bar(
        bottom_10_cities,
        x="city",
        y="price",
        title=f"Bottom 10 Cities in {selected_state_bottom_10} by Median Price",
        labels={"price": "Median Price ($)", "city": "City"},
        color="price",
        color_continuous_scale="Reds"
    )
    fig.update_layout(
    xaxis_tickfont_color='black',  
    yaxis_tickfont_color='black',  
    xaxis_title_font_color='black',
    yaxis_title_font_color='black',
    legend_font_color='black',
    coloraxis_colorbar_tickfont_color='black', 
    coloraxis_colorbar_title_font_color='black'
    )
    st.plotly_chart(fig)
    
    #5. Correlation Heatmap
    st.subheader("Correlation Matrix Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = data[['bed', 'bath', 'house_size', 'price']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)
    
