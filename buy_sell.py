import ccxt
from binance.client import Client

apiKey = ''
secretKey = ''
binance_account_info = {
'apiKey': apiKey,
'secret': secretKey
}
# Initialize the CCXT exchange instance
exchange = ccxt.binance({
    'apiKey': apiKey,
    'secret': secretKey,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',  # Set the default trade type to 'future'
    }
})

client = Client(binance_account_info['apiKey'], binance_account_info['secret'])

def open_buy_position(symbol, amount):
    try:
        # Define order parameters
        order_params = {
            'symbol': symbol,
            'side': 'buy',
            'type': 'market',
            'amount': amount, # Good 'Til Cancelled
        }

        # Place the buy order
        order = exchange.create_order(**order_params)

        return order
    except Exception as e:
        return f"Error placing buy order: {e}"

def open_sell_position(symbol, amount):
    try:
        # Define order parameters
        order_params = {
            'symbol': symbol,
            'side': 'sell',
            'type': 'market',
            'amount': amount,
        }
        wallets = exchange.fetch_balance()
        print(wallets['USDT'])
        # Place the sell order
        order = exchange.create_order(**order_params)

        return order
    except Exception as e:
        return f"Error placing sell order: {e}"

def close_position(symbol, amount, type):
    try:
        # Define order parameters
        order_params = {
            'symbol': symbol,
            'side': type,  # Close a long position by selling
            'type': 'market',  # Market order to close
            'amount': amount,
        }

        # Place the sell order to close the position
        order = exchange.create_order(**order_params)

        return order
    except Exception as e:
        return f"Error closing position: {e}"


def get_balance(symbol, amount):
    try:
        # Define order parameters
        order_params = {
            'symbol': symbol,
            'side': 'sell',
            'type': 'market',
            # 'amount': amount,
        }
        wallets = exchange.fetch_balance()
        print(wallets['USDT'])
        return wallets['USDT']
    except Exception as e:
        print(e)
        return {"free": amount, "used":amount, "total":amount}


def get_profit(symbol):
    # Fetch account information
    account_info = client.get_account()
    for asset in account_info['balances']:
        # Find the specified trading pair in the account information
        if asset['asset'] == symbol[:-4]:
            asset_balance = float(asset['free']) + float(asset['locked'])
            # Fetch trades for the specified trading pair
            trades = client.get_my_trades(symbol=symbol, limit=5)

            pnl = 0.0
            for trade in trades:
                if trade['isBuyer']:
                    # Bought LPTUSDT
                    pnl -= float(trade['quoteQty'])
                else:
                    # Sold LPTUSDT
                    pnl += float(trade['quoteQty'])

            return pnl

    return None




