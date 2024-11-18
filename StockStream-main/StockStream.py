
from matplotlib.pyplot import axis
import matplotlib.pyplot as plt
import streamlit as st  # streamlit library
import pandas as pd  # pandas library
import requests
import plotly.express as px
import numpy as np
import yfinance as yf  # yfinance library
import datetime  # datetime library
from datetime import date
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
from newspaper import Article
import io
import nltk
from PIL import Image
from plotly import graph_objs as go  # plotly library
from plotly.subplots import make_subplots
from prophet import Prophet  # prophet library
# plotly library for prophet model plotting
from prophet.plot import plot_plotly
import time  # time library
from streamlit_option_menu import option_menu  # select_options library
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
def fetch_business_news():
    site = 'https://news.google.com/news/rss/headlines/section/topic/BUSINESS'
    op = urlopen(site)
    rd = op.read()
    op.close()
    sp_page = soup(rd, 'xml')
    news_list = sp_page.find_all('item')
    return news_list


def fetch_news_search_topic(topic):
    site = 'https://news.google.com/rss/search?q={}'.format(topic)
    op = urlopen(site)
    rd = op.read()
    op.close()
    sp_page = soup(rd, 'xml')
    news_list = sp_page.find_all('item')
    return news_list
# Fetch real-time data
def fetch_live_prices():
    # Tickers for desired commodities and indices
    tickers = {
        "Gold (GC=F)": ("GC=F", "USD", "per oz"),
        "Silver (SI=F)": ("SI=F", "USD", "per oz"),
        "Crude Oil (CL=F)": ("CL=F", "USD", "per barrel"),
        "Nifty 50 (^NSEI)": ("^NSEI", "INR", "points"),
        "Sensex (^BSESN)": ("^BSESN", "INR", "points"),
        "USD/INR (USDINR=X)": ("USDINR=X", "INR", "per USD"),
    }
    
    live_data = {}
    for name, details in tickers.items():
        ticker, currency, unit = details  # Explicit unpacking
        try:
            stock = yf.Ticker(ticker)
            price = stock.history(period="1d")['Close'].iloc[-1]
            live_data[name] = f"{currency} {price:,.2f} {unit}"
        except Exception as e:
            live_data[name] = "Data Unavailable"
    return live_data

def fetch_news_poster(poster_link):
    try:
        u = urlopen(poster_link)
        raw_data = u.read()
        image = Image.open(io.BytesIO(raw_data))
        st.image(image, use_column_width=True)
    except:
        image = Image.open('./Meta/no_image.jpg')
        st.image(image, use_column_width=True)


def display_news(list_of_news, news_quantity):
    c = 0
    for news in list_of_news:
        c += 1
        st.write(f"**({c}) {news.title.text}**")  # News title
        news_data = Article(news.link.text)
        try:
            news_data.download()
            news_data.parse()
            news_data.nlp()
        except Exception as e:
            st.error(e)
        fetch_news_poster(news_data.top_image)  # Display poster image
        with st.expander(news.title.text):  # Expandable section
            st.markdown(
                f"""<h6 style='text-align: justify;'>{news_data.summary}</h6>""",
                unsafe_allow_html=True,
            )  # News summary
            # Clickable link
            st.markdown(f"[Read full article here]({news.link.text})", unsafe_allow_html=True)
        st.success(f"Published Date: {news.pubDate.text}")
        if c >= news_quantity:
            break
def run():
    st.title("BizNews💼: Business & Finance News")
    image = Image.open('./Meta/newspaper.png')

    col1, col2, col3 = st.columns([3, 5, 3])

    with col1:
        st.write("")

    with col2:
        st.image(image, use_column_width=False)

    with col3:
        st.write("")

    category = ['--Select--', 'Trending📈 Business News', 'Live Market Prices📊']
    cat_op = st.selectbox('Select your Category', category)

    if cat_op == category[0]:
        st.warning('Please select a Category!')
    elif cat_op == category[1]:
        st.subheader("✅ Here are the latest 📈 Business News for you")
        no_of_news = st.slider('Number of News:', min_value=5, max_value=25, step=1)
        news_list = fetch_business_news()
        display_news(news_list, no_of_news)
    elif cat_op == category[2]:
        st.subheader("📊 Live Market Prices for Key Commodities & Indexes")
        prices = fetch_live_prices()
        for key, value in prices.items():
            st.metric(label=key, value=value)

def add_meta_tag():
    meta_tag = """
        <head>
            <meta name="google-site-verification" content="QBiAoAo1GAkCBe1QoWq-dQ1RjtPHeFPyzkqJqsrqW-s" />
        </head>
    """
    st.markdown(meta_tag, unsafe_allow_html=True)

# Main code
add_meta_tag()

# Sidebar Section Starts Here
today = date.today()  # today's date
st.write('''# StockStream ''')  # title
st.sidebar.image("Images/StockStreamLogo1.png", width=250,
                 use_column_width=False)  # logo
st.sidebar.write('''# StockStream ''')

with st.sidebar: 
        selected = option_menu("Utilities", ["Stocks Performance Comparison", "Real-Time Stock Price", "Stock Prediction","News", 'About'])

start = st.sidebar.date_input(
    'Start', datetime.date(2022, 1, 1))  # start date input
end = st.sidebar.date_input('End', datetime.date.today())  # end date input
# Sidebar Section Ends Here

# read csv file
stock_df = pd.read_csv("StockStreamTickersData.csv")

# Stock Performance Comparison Section Starts Here
if(selected == 'Stocks Performance Comparison'):  # if user selects 'Stocks Performance Comparison'
    st.subheader("Stocks Performance Comparison")
    tickers = stock_df["Company Name"]
    # dropdown for selecting assets
    dropdown = st.multiselect('Pick your assets', tickers)

    with st.spinner('Loading...'):  # spinner while loading
        time.sleep(2)
        # st.success('Loaded')

    dict_csv = pd.read_csv('StockStreamTickersData.csv', header=None, index_col=0).to_dict()[1]  # read csv file
    symb_list = []  # list for storing symbols
    for i in dropdown:  # for each asset selected
        val = dict_csv.get(i)  # get symbol from csv file
        symb_list.append(val)  # append symbol to list

    def relativeret(df):  # function for calculating relative return
        rel = df.pct_change()  # calculate relative return
        cumret = (1+rel).cumprod() - 1  # calculate cumulative return
        cumret = cumret.fillna(0)  # fill NaN values with 0
        return cumret  # return cumulative return

    if len(dropdown) > 0:  # if user selects atleast one asset
        df = relativeret(yf.download(symb_list, start, end))[
            'Adj Close']  # download data from yfinance
        # download data from yfinance
        raw_df = relativeret(yf.download(symb_list, start, end))
        raw_df.reset_index(inplace=True)  # reset index

        closingPrice = yf.download(symb_list, start, end)[
            'Adj Close']  # download data from yfinance
        volume = yf.download(symb_list, start, end)['Volume']
        
        st.subheader('Raw Data {}'.format(dropdown))
        st.write(raw_df)  # display raw data
        chart = ('Line Chart', 'Area Chart', 'Bar Chart')  # chart types
        # dropdown for selecting chart type
        dropdown1 = st.selectbox('Pick your chart', chart)
        with st.spinner('Loading...'):  # spinner while loading
            time.sleep(2)

        st.subheader('Relative Returns {}'.format(dropdown))
                
        if (dropdown1) == 'Line Chart':  # if user selects 'Line Chart'
            st.line_chart(df)  # display line chart
            # display closing price of selected assets
            st.write("### Closing Price of {}".format(dropdown))
            st.line_chart(closingPrice)  # display line chart

            # display volume of selected assets
            st.write("### Volume of {}".format(dropdown))
            st.line_chart(volume)  # display line chart

        elif (dropdown1) == 'Area Chart':  # if user selects 'Area Chart'
            st.area_chart(df)  # display area chart
            # display closing price of selected assets
            st.write("### Closing Price of {}".format(dropdown))
            st.area_chart(closingPrice)  # display area chart

            # display volume of selected assets
            st.write("### Volume of {}".format(dropdown))
            st.area_chart(volume)  # display area chart

        elif (dropdown1) == 'Bar Chart':  # if user selects 'Bar Chart'
            st.bar_chart(df)  # display bar chart
            # display closing price of selected assets
            st.write("### Closing Price of {}".format(dropdown))
            st.bar_chart(closingPrice)  # display bar chart

            # display volume of selected assets
            st.write("### Volume of {}".format(dropdown))
            st.bar_chart(volume)  # display bar chart

        else:
            st.line_chart(df, width=1000, height=800,
                          use_container_width=False)  # display line chart
            # display closing price of selected assets
            st.write("### Closing Price of {}".format(dropdown))
            st.line_chart(closingPrice)  # display line chart

            # display volume of selected assets
            st.write("### Volume of {}".format(dropdown))
            st.line_chart(volume)  # display line chart

    else:  # if user doesn't select any asset
        st.write('Please select atleast one asset')  # display message
# Stock Performance Comparison Section Ends Here
    
# Real-Time Stock Price Section Starts Here
elif(selected == 'Real-Time Stock Price'):  # if user selects 'Real-Time Stock Price'
    st.subheader("Real-Time Stock Price")
    tickers = stock_df["Company Name"]  # get company names from csv file
    # dropdown for selecting company
    a = st.selectbox('Pick a Company', tickers)

    with st.spinner('Loading...'):  # spinner while loading
            time.sleep(2)

    dict_csv = pd.read_csv('StockStreamTickersData.csv', header=None, index_col=0).to_dict()[1]  # read csv file
    symb_list = []  # list for storing symbols

    val = dict_csv.get(a)  # get symbol from csv file
    symb_list.append(val)  # append symbol to list

    if "button_clicked" not in st.session_state:  # if button is not clicked
        st.session_state.button_clicked = False  # set button clicked to false

    def callback():  # function for updating data
        # if button is clicked
        st.session_state.button_clicked = True  # set button clicked to true
    if (
        st.button("Search", on_click=callback)  # button for searching data
        or st.session_state.button_clicked  # if button is clicked
    ):
        if(a == ""):  # if user doesn't select any company
            st.write("Click Search to Search for a Company")
            with st.spinner('Loading...'):  # spinner while loading
             time.sleep(2)
        else:  # if user selects a company
            # download data from yfinance
            data = yf.download(symb_list, start=start, end=end)
            data.reset_index(inplace=True)  # reset index
            st.subheader('Raw Data of {}'.format(a))  # display raw data
            st.write(data)  # display data

            def plot_raw_data():  # function for plotting raw data
                fig = go.Figure()  # create figure
                fig.add_trace(go.Scatter(  # add scatter plot
                    x=data['Date'], y=data['Open'], name="stock_open"))  # x-axis: date, y-axis: open
                fig.add_trace(go.Scatter(  # add scatter plot
                    x=data['Date'], y=data['Close'], name="stock_close"))  # x-axis: date, y-axis: close
                fig.layout.update(  # update layout
                    title_text='Line Chart of {}'.format(a) , xaxis_rangeslider_visible=True)  # title, x-axis: rangeslider
                st.plotly_chart(fig)  # display plotly chart

            def plot_candle_data():  # function for plotting candle data
                fig = go.Figure()  # create figure
                fig.add_trace(go.Candlestick(x=data['Date'],  # add candlestick plot
                                             # x-axis: date, open
                                             open=data['Open'],
                                             high=data['High'],  # y-axis: high
                                             low=data['Low'],  # y-axis: low
                                             close=data['Close'], name='market data'))  # y-axis: close
                fig.update_layout(  # update layout
                    title='Candlestick Chart of {}'.format(a),  # title
                    yaxis_title='Stock Price',  # y-axis: title
                    xaxis_title='Date')  # x-axis: title
                st.plotly_chart(fig)  # display plotly chart

            chart = ('Candle Stick', 'Line Chart')  # chart types
            # dropdown for selecting chart type
            dropdown1 = st.selectbox('Pick your chart', chart)
            with st.spinner('Loading...'):  # spinner while loading
             time.sleep(2)
            if (dropdown1) == 'Candle Stick':  # if user selects 'Candle Stick'
                plot_candle_data()  # plot candle data
            elif (dropdown1) == 'Line Chart':  # if user selects 'Line Chart'
                plot_raw_data()  # plot raw data
            else:  # if user doesn't select any chart
                plot_candle_data()  # plot candle data

# Real-Time Stock Price Section Ends Here

# Stock Price Prediction Section Starts Here
elif(selected == 'Stock Prediction'):  # if user selects 'Stock Prediction'
    stocks_csv_path = 'StockStreamTickersData.csv'  # Replace with your CSV file path
    stocks_df = pd.read_csv(stocks_csv_path)

    # Check if required columns exist in the CSV
    if 'Company Name' not in stocks_df.columns or 'Symbol' not in stocks_df.columns:
        st.error("The CSV file must have 'Stock Name' and 'Stock Code' columns.")
    else:
        # Map stock names to their codes
        stock_dict = dict(zip(stocks_df['Company Name'], stocks_df['Symbol']))

        # Add a dropdown to select a stock
        selected_stock_name = st.selectbox(
            "Select a stock to analyze",
            options=list(stock_dict.keys()),
            format_func=lambda x: f"{x} ({stock_dict[x]})"
        )

        # Retrieve the selected stock code
    stock_code = stock_dict[selected_stock_name]
    st.write(f"Selected Stock: {selected_stock_name} ({stock_code})")
    # Date range for fetching data
    start = '2010-01-01'
    end = '2023-12-31'

    # Load pre-trained model
    model = load_model('keras_model.h5')

    # Fetch stock data
    st.subheader(f"Analysis for {selected_stock_name} ({stock_code})")
    df = yf.download(stock_code, start=start, end=end)

    if df.empty:
        st.write(f"No data found for {stock_code}. Please check the ticker symbol.")
    else:
        # Display basic stats
        st.write(df.describe())

        # Plot Closing Prices
        st.subheader('Closing Price Vs Time')
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Closing Price')
        plt.legend()
        st.pyplot(fig)

        # Add Moving Averages
        st.subheader('Closing Price Vs Time with 100MA & 200MA')
        ma100 = df['Close'].rolling(100).mean()
        ma200 = df['Close'].rolling(200).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Closing Price')
        plt.plot(ma100, label='100-Day MA', color='red')
        plt.plot(ma200, label='200-Day MA', color='green')
        plt.legend()
        st.pyplot(fig)

        # Prepare data for prediction
        data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

        # Scale the data (Use training scaler fitted on AAPL)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_training = scaler.fit_transform(data_training)
        scaled_testing = scaler.transform(pd.concat([data_training.tail(100), data_testing]))

        x_test = []
        y_test = []
        for i in range(100, scaled_testing.shape[0]):
            x_test.append(scaled_testing[i-100:i])
            y_test.append(scaled_testing[i, 0])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # Make predictions
        y_predicted = model.predict(x_test)

        # Rescale predictions to original values
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Plot Original vs Predicted
        st.subheader('Original Vs Prediction')
        fig = plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Original Price', color='blue')
        plt.plot(y_predicted, label='Predicted Price', color='orange')
        plt.legend()
        st.pyplot(fig)
        dates = data_testing.index[100:]
        min_length = min(len(dates), len(y_test), len(y_predicted))
        dates = dates[:min_length]
        y_test = y_test[:min_length]
        y_predicted = y_predicted[:min_length]

        # Create DataFrame for comparison
        comparison_df = pd.DataFrame({
            'Date': dates,
            'Actual Price': y_test.flatten(),
            'Predicted Price': y_predicted.flatten(),
        })
        comparison_df['Percentage Error'] = abs(
            (comparison_df['Actual Price'] - comparison_df['Predicted Price']) / comparison_df['Actual Price'] * 100
        )

        # Display the comparison table
        st.subheader('Prediction vs Actual Prices')
        st.write(comparison_df)
        st.subheader('Actual vs Predicted Price Chart')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(comparison_df['Date'], comparison_df['Actual Price'], label='Actual Price', color='blue')
        ax.plot(comparison_df['Date'], comparison_df['Predicted Price'], label='Predicted Price', color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('Actual vs Predicted Price Over Time')
        ax.legend()

        # Rotate dates for better readability
        plt.xticks(rotation=45)
        st.pyplot(fig)
        from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
        mape = mean_absolute_percentage_error(y_test, y_predicted) * 100
        mse = mean_squared_error(y_test, y_predicted)
        rmse = np.sqrt(mse)

        #Dividend Analysis 
        stock = yf.Ticker(stock_code)
        fundamentals = {
        "Previous Close": stock.info.get("previousClose"),
        "Open": stock.info.get("open"),
        "52-Week High": stock.info.get("fiftyTwoWeekHigh"),
        "52-Week Low": stock.info.get("fiftyTwoWeekLow"),
        "Market Cap": stock.info.get("marketCap"),
        "Dividend Yield (%)": stock.info.get("dividendYield") * 100 if stock.info.get("dividendYield") else "N/A",
        "P/E Ratio (TTM)": stock.info.get("trailingPE"),
        "P/B Ratio": stock.info.get("priceToBook"),
        "EPS (TTM)": stock.info.get("trailingEps"),
        "Beta": stock.info.get("beta"),
        "Profit Margins (%)": stock.info.get("profitMargins") * 100 if stock.info.get("profitMargins") else "N/A",}

        fundamentals_df = pd.DataFrame(fundamentals.items(), columns=["Parameter", "Value"])
        fundamentals_df = pd.DataFrame(fundamentals.items(), columns=["Parameter", "Value"])

    # Display fundamentals table
        st.subheader(f"Fundamental Parameters of {stock_code}")
        st.write(fundamentals_df)

        st.subheader('Model Accuracy Metrics')
        st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Stock Price Prediction Section Ends Here
elif (selected == 'News'):
    run()
elif(selected == 'About'):
    st.subheader("About")
    
    st.markdown("""
        <style>
    .big-font {
        font-size:25px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.image("Images/R.png", caption="StockStream - Visualize, Predict, and Analyze", width=300)
    st.markdown('<p class="big-font">StockStream is a web application that allows users to visualize Stock Performance Comparison, Real-Time Stock Prices and Stock Price Prediction. This application is built by Daksh Jain , Sehal Saxena and Mayer Goyal students of Jaypee Institute of Information Technology as their minor project .  </p>', unsafe_allow_html=True)
 
 