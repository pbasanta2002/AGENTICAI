def getStockPrice(stock_symbol, ALPHA_VANTAGE_API_KEY, logger):
    import requests
    # Call Alpha Vantage API
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Alpha Vantage API response for {stock_symbol}: {data.keys()}")

        if "Time Series (Daily)" in data:
            latest_date = list(data["Time Series (Daily)"].keys())[0]
            stock_data = data["Time Series (Daily)"][latest_date]
            close_price = stock_data["4. close"]
            message = f"The latest closing price for {stock_symbol} is ${close_price} (as of {latest_date})."

            # # Add risk tolerance advice
            # risk_tolerance = user_profile.get('risk tolerance', 'unknown')
            # risk_prompt = (
            #     f"Provide a brief note on investing in {stock_symbol} tailored to a user with {risk_tolerance} risk tolerance. "
            #     f"Keep it clear and empathetic."
            # )
            # risk_response = await llm.ainvoke(risk_prompt)
            # message += f"\n{risk_response.content.strip()}"
        elif "Error Message" in data:
            message = f"Error from Alpha Vantage: {data['Error Message']}. Please check the stock symbol or try again later."
            logger.error(f"Alpha Vantage error for {stock_symbol}: {data['Error Message']}")
        elif "Note" in data and "rate limit" in data["Note"].lower():
            message = "Alpha Vantage API rate limit exceeded. Please try again in a minute."
            logger.warning(f"Rate limit exceeded for {stock_symbol}: {data['Note']}")
        else:
            message = f"No data available for {stock_symbol}. Please check the symbol or try again later."
            logger.error(f"No time series data for {stock_symbol}: {data}")
    except requests.RequestException as e:
        message = f"Error fetching data for {stock_symbol}: {str(e)}. Please try again later."
        logger.error(f"Request error for {stock_symbol}: {str(e)}")
    return close_price, message