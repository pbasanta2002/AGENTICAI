def getStockPrice(stock_symbol):
    import yfinance as yf
    tckr = yf.Ticker(stock_symbol)
    # Get fast info and the last price
    actual_price = tckr.fast_info.last_price # 'last_price' is typically the most recent trading price
    close_price = tckr.fast_info.previous_close
    # print(tckr.fast_info)
    message = f"The latest closing price for {stock_symbol} is ${close_price}."
    return close_price, message

# symbol = 'TSLA'
# latestPrice, previousClosePrice = getStockPrice(symbol)
# print(f"The latest actual price for {symbol} is: {latestPrice}")
# print(f"The previous closing price for {symbol} was: {previousClosePrice}")