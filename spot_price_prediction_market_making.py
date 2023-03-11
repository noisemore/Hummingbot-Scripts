import logging
from decimal import Decimal
from typing import Dict

import pandas as pd
import pandas_ta as ta  # noqa: F401
from hummingbot.connector.connector_base import ConnectorBase
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from hummingbot.connector.utils import split_hb_trading_pair
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    BuyOrderCreatedEvent,
    MarketOrderFailureEvent,
    OrderCancelledEvent,
    OrderFilledEvent,
    SellOrderCompletedEvent,
    SellOrderCreatedEvent,
)
from hummingbot.core.event.events import OrderType
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


# ---------------------------------------------------------------------------------------------------#

def interval_in_sec(_data_interval):
    data_interval = int(14400)
    if str("m") in _data_interval:
        data_interval = _data_interval.replace(str("m"), str(""))
        data_interval = int(data_interval)
        data_interval = int(data_interval * 60)
    elif "h" in _data_interval:
        data_interval = _data_interval.replace(str("h"), str(""))
        data_interval = int(data_interval)
        data_interval = int(data_interval * 60 * 60)
    elif "d" in _data_interval:
        data_interval = _data_interval.replace(str("d"), str(""))
        data_interval = int(data_interval)
        data_interval = int(data_interval * 60 * 60 * 24)
    elif "w" in _data_interval:
        data_interval = _data_interval.replace(str("w"), str(""))
        data_interval = int(data_interval)
        data_interval = int(data_interval * 60 * 60 * 24 * 7)
    return data_interval


class spot_price_prediction_market_making(ScriptStrategyBase):
    # Connector Parameters
    connector_name = str("binance")
    data_connector = str("binance")
    market_pair = str("RLC-USDT")
    coin_base, coin_quote = split_hb_trading_pair(market_pair)
    markets = {connector_name: {market_pair}}

    # Trend Parameters
    trend_interval = str("4h")
    trend_candles = CandlesFactory.get_candle(connector=data_connector,
                                              trading_pair=market_pair,
                                              interval=trend_interval,
                                              max_records=60)
    # Set Vortex config
    vtx_length = int(14)

    # Set SuperTrend config
    st_length = int(6)
    st_multiplier = float(2.618)

    # Set ADX config
    adx_length = int(14)

    # Set Dataframe config
    trend_candles_df = pd.DataFrame()
    last_trend = bool(True)
    action_signal = str("")
    have_data = bool(False)

    # Analysis Parameters
    analysis_interval = str("1m")
    analysis_candles = CandlesFactory.get_candle(connector=data_connector,
                                                 trading_pair=market_pair,
                                                 interval=analysis_interval,
                                                 max_records=60)

    # Set price prediction config
    mom_len = int(10)
    mfi_len = int(14)
    atr_len = int(14)
    atr_prediction = Decimal("0.00")
    analysis_candles_df = pd.DataFrame()

    # Set build inventory config
    # build_ordered_ts = int(0)
    # build_interval = int(10)

    # Set order config
    # last_ordered_ts = int(0)
    # order_interval = int(1)
    have_inventory = bool(False)
    mid_price = Decimal("0.00")
    total_budget = Decimal("100")
    order_usd = Decimal("50")
    u_buy_atr_multiplier = Decimal("2")
    u_sell_atr_multiplier = Decimal("2.5")
    d_buy_atr_multiplier = Decimal("2.5")
    d_sell_atr_multiplier = Decimal("2")
    stop_loss = Decimal('0.1')
    min_order_size = Decimal("10.0")
    run_the_bot = bool(True)

    def __init__(self, connectors: Dict[str, ConnectorBase]):
        # Is necessary to start the Candles Feed.
        super().__init__(connectors)
        self.trend_candles.start()
        self.analysis_candles.start()

    @property
    def all_candles_ready(self):
        """
        Checks if the candlesticks are full.
        :return:
        """
        return all([self.trend_candles.is_ready, self.analysis_candles.is_ready])

    @property
    def current_price(self):
        return self.connectors[self.connector_name].get_mid_price(self.market_pair)

    def on_tick(self):

        if self.run_the_bot is True:

            # ========================================== Technical Analysis Start =========================================#

            # Get initial trend data and ATR prediction
            if self.have_data is False:
                self.trend_candles_df, self.last_trend, self.action_signal = self.get_trend_signal()
                self.analysis_candles_df, self.atr_prediction = self.get_atr_prediction()
                self.have_data = True
            else:
                if self.current_timestamp == (
                        (self.current_timestamp // interval_in_sec(self.trend_interval)) * interval_in_sec(
                    self.trend_interval)):
                    self.trend_candles_df, self.last_trend, self.action_signal = self.get_trend_signal()
                if self.current_timestamp == (
                        (self.current_timestamp // interval_in_sec(self.analysis_interval)) * interval_in_sec(
                    self.analysis_interval)):
                    self.analysis_candles_df, self.atr_prediction = self.get_atr_prediction()

            # ========================================== Technical Analysis End ===========================================#

            # ========================================== Order Management Start ===========================================#

            if self.get_stop_loss():
                if self.get_have_inventory():
                    # If current value is lower than stop loss, sell all
                    self.cancel_all_orders()
                    self.sell(
                        connector_name=self.connector_name,
                        trading_pair=self.market_pair,
                        amount=self.connectors[self.connector_name].get_balance(self.coin_base),
                        order_type=OrderType.MARKET,
                        price=self.current_price
                    )
                    self.run_the_bot = False
            else:
                buy_atr_multiplier = self.u_buy_atr_multiplier if self.last_trend else self.d_buy_atr_multiplier
                sell_atr_multiplier = self.u_sell_atr_multiplier if self.last_trend else self.d_sell_atr_multiplier

                if self.mid_price != self.current_price:
                    self.cancel_all_orders()

                    buy_target_price = self.current_price - (self.atr_prediction * buy_atr_multiplier)
                    sell_target_price = self.current_price + (self.atr_prediction * sell_atr_multiplier)
                    buy_order_amount = self.buy_amount(buy_target_price)
                    sell_order_amount = self.sell_amount()

                    self.buy(
                        connector_name=self.connector_name,
                        trading_pair=self.market_pair,
                        amount=buy_order_amount,
                        order_type=OrderType.LIMIT,
                        price=buy_target_price
                    )
                    self.sell(
                        connector_name=self.connector_name,
                        trading_pair=self.market_pair,
                        amount=sell_order_amount,
                        order_type=OrderType.LIMIT,
                        price=sell_target_price
                    )

                    self.mid_price = self.current_price

            # ========================================== Order Management End =============================================#

    def on_stop(self):
        """
        Without this functionality, the network iterator will continue running forever after stopping the strategy
        That's why is necessary to introduce this new feature to make a custom stop with the strategy.
        :return:
        """
        self.trend_candles.stop()
        self.analysis_candles.stop()

    def cancel_all_orders(self):
        """
        Cancel all orders from the bot
        """
        for order in self.get_active_orders(connector_name=self.connector_name):
            self.cancel(self.connector_name, order.trading_pair, order.client_order_id)

    def buy_amount(self, buy_target_price):
        if self.connectors[self.connector_name].get_balance(self.coin_quote) >= Decimal(self.order_usd):
            buy_amount = self.order_usd / buy_target_price
        elif self.min_order_size <= self.connectors[self.connector_name].get_balance(self.coin_quote) < Decimal(
                self.order_usd):
            buy_amount = self.connectors[self.connector_name].get_balance(self.coin_quote)
        else:
            buy_amount = Decimal('0.0')

        return buy_amount

    def sell_amount(self):
        if Decimal(self.min_order_size) <= self.connectors[self.connector_name].get_balance(
                self.coin_base) * self.current_price <= Decimal(self.order_usd):
            sell_order_amount = self.connectors[self.connector_name].get_balance(self.coin_base)
        elif Decimal(self.order_usd) <= self.connectors[self.connector_name].get_balance(
                self.coin_base) * self.current_price:
            sell_order_amount = (self.connectors[self.connector_name].get_balance(self.coin_base) * self.get_price(
                self.market_pair, False)) / self.order_usd
        elif self.connectors[self.connector_name].get_balance(self.coin_quote) <= Decimal(self.order_usd):
            sell_order_amount = self.connectors[self.connector_name].get_balance(self.coin_base) / Decimal("2")
        else:
            sell_order_amount = Decimal('0.0')

        return sell_order_amount

    def build_inventory(self):
        """
        Build inventory
        """
        if self.have_inventory is False:
            if (self.connectors[self.connector_name].get_balance(
                    self.coin_base) * self.current_price) >= self.total_budget / Decimal("2"):
                self.have_inventory = True
                self.logger().info("Inventory already built!")
            else:
                if self.build_ordered_ts < (self.current_timestamp - self.build_interval):
                    order_amount = self.order_usd / self.connectors[self.connector_name].get_price(self.market_pair,
                                                                                                   True)
                    self.logger().info("Building inventory...")
                    self.cancel_all_orders()
                    self.buy(connector_name=self.connector_name,
                             trading_pair=self.market_pair,
                             amount=order_amount,
                             order_type=OrderType.LIMIT,
                             price=self.connectors[self.connector_name].get_price(self.market_pair, True))
                    self.build_ordered_ts = self.current_timestamp

    def get_have_inventory(self):
        return bool(self.connectors[self.connector_name].get_balance(self.coin_base) > Decimal("0"))

    def get_stop_loss(self):
        """
        Stop loss check
        """
        current_value = self.connectors[self.connector_name].get_balance(self.coin_quote) + self.connectors[
            self.connector_name].get_balance(self.coin_base) * self.current_price
        stop_loss_value = self.total_budget * (Decimal('1') - self.stop_loss)

        return bool(current_value < stop_loss_value)

    def get_trend_signal(self):
        trend_candles_df = self.trend_candles.candles_df
        trend_candles_df.drop(axis=1, columns={'quote_asset_volume', 'n_trades', 'taker_buy_base_volume',
                                               'taker_buy_quote_volume'}, inplace=True)

        # VTX Indicator
        vtx_props = f"_{self.vtx_length}"
        trend_candles_df.ta.vortex(append=True)
        trend_candles_df = trend_candles_df.rename({f'VTXP{vtx_props}': 'VTXP', f'VTXM{vtx_props}': 'VTXM'},
                                                   axis='columns')
        trend_candles_df['VTX_uptrend'] = trend_candles_df.apply(lambda row: row['VTXP'] > row['VTXM'],
                                                                 axis=1).astype(bool)

        # SuperTrend Indicator
        st_props = f"_{self.st_length}_{self.st_multiplier}"
        trend_candles_df.ta.supertrend(length=self.st_length, multiplier=self.st_multiplier, append=True)
        trend_candles_df = trend_candles_df.rename({f'SUPERTd{st_props}': 'ST_uptrend'}, axis='columns')
        trend_candles_df['ST_uptrend'] = trend_candles_df['ST_uptrend'].replace({1: True, -1: False}).astype(bool)

        # ADX Indicator
        adx_props = f"_{self.adx_length}"
        trend_candles_df.ta.adx(length=self.adx_length, append=True)
        trend_candles_df = trend_candles_df.rename(
            {f'ADX{adx_props}': 'ADX', f'DMP_{self.adx_length}': 'DMP', f'DMN_{self.adx_length}': 'DMN'},
            axis='columns')
        trend_candles_df['ADX_uptrend'] = trend_candles_df.apply(lambda row: row['DMP'] > row['DMN'],
                                                                 axis=1).astype(bool)

        trend_candles_df['uptrend'] = (trend_candles_df['VTX_uptrend'].astype(int) +
                                       trend_candles_df['ST_uptrend'].astype(int) +
                                       trend_candles_df['ADX_uptrend'].astype(int)) >= 2

        trend_candles_df.drop(axis=1, columns={'VTXP', 'VTXM', f'SUPERT{st_props}', f'SUPERTl{st_props}',
                                               f'SUPERTs{st_props}', 'ADX', 'DMP', 'DMN', 'VTX_uptrend',
                                               'ST_uptrend', 'ADX_uptrend'}, inplace=True)
        trend_candles_df["timestamp"] = pd.to_datetime(trend_candles_df["timestamp"], unit="ms")

        pre_trend = bool(trend_candles_df.iloc[-2]['uptrend'])
        last_trend = bool(trend_candles_df.iloc[-1]['uptrend'])

        if last_trend is True and pre_trend is False:
            action_signal = str("Long")
        elif last_trend is False and pre_trend is True:
            action_signal = str("Short")
        else:
            action_signal = str("No Action")

        return trend_candles_df, last_trend, action_signal

    def get_atr_prediction(self):
        # Input candles
        analysis_candles_df = self.analysis_candles.candles_df.copy()
        analysis_candles_df.drop(axis=1,
                                 columns={
                                     'quote_asset_volume',
                                     'n_trades',
                                     'taker_buy_base_volume',
                                     'taker_buy_quote_volume'},
                                 inplace=True)
        # Calculate Trading Volume, HLC3, and True Range Indicators
        analysis_candles_df['tr'] = analysis_candles_df.ta.true_range(high=analysis_candles_df['high'],
                                                                      low=analysis_candles_df['low'],
                                                                      close=analysis_candles_df['close'], inplace=True)

        analysis_candles_df['mom'] = analysis_candles_df.ta.mom(close=analysis_candles_df['close'],
                                                                length=self.mom_len, inplace=True)

        analysis_candles_df['mfi'] = analysis_candles_df.ta.mfi(high=analysis_candles_df['high'],
                                                                low=analysis_candles_df['low'],
                                                                close=analysis_candles_df['close'],
                                                                volume=analysis_candles_df['volume'],
                                                                length=self.mfi_len, inplace=True)

        analysis_candles_df['atr'] = analysis_candles_df.ta.atr(high=analysis_candles_df['high'],
                                                                low=analysis_candles_df['low'],
                                                                close=analysis_candles_df['close'],
                                                                length=self.atr_len, inplace=True)

        # Shift the data in the High and Low columns.
        analysis_candles_df['atr_1'] = analysis_candles_df['atr'].shift(1)
        analysis_candles_df.dropna(inplace=True)

        # Define the feature and target variables.
        features = analysis_candles_df[['volume', 'high', 'low', 'close', 'tr', 'mom', 'mfi']]
        target = analysis_candles_df['atr_1']

        features_train, features_test, target_train, target_test = train_test_split(features, target,
                                                                                    test_size=0.35, random_state=0)

        # Create a linear regression model and train it.
        lr = LinearRegression()
        lr.fit(features_train, target_train)

        # Make predictions on the test set.
        target_pred = lr.predict(features_test)

        # Calculate the root mean squared error (RMSE).
        # mse = mean_squared_error(targets_test, targets_pred)
        # rmse = mse ** 0.5

        features = features.dropna()
        analysis_candles_df['target_pred'] = lr.predict(features)
        analysis_candles_df["timestamp"] = pd.to_datetime(analysis_candles_df["timestamp"], unit="ms")
        atr_pred = Decimal(str(analysis_candles_df['target_pred'].iloc[-1]))
        analysis_candles_df.drop(axis=1, columns={'target_pred', 'tr', 'mom', 'mfi', 'atr_1'}, inplace=True)

        return analysis_candles_df, atr_pred

    def did_create_buy_order(self, event: BuyOrderCreatedEvent):
        """
        Method called when the connector notifies a buy order has been created
        """
        self.logger().info(logging.INFO, f"The buy order {event.order_id} has been created")

    def did_create_sell_order(self, event: SellOrderCreatedEvent):
        """
        Method called when the connector notifies a sell order has been created
        """
        self.logger().info(logging.INFO, f"The sell order {event.order_id} has been created")

    def did_fill_order(self, event: OrderFilledEvent):
        """
        Method called when the connector notifies that an order has been partially or totally filled (a trade happened)
        """
        self.logger().info(logging.INFO, f"The order {event.order_id} has been filled")

    def did_fail_order(self, event: MarketOrderFailureEvent):
        """
        Method called when the connector notifies an order has failed
        """
        self.logger().info(logging.INFO, f"The order {event.order_id} failed")

    def did_cancel_order(self, event: OrderCancelledEvent):
        """
        Method called when the connector notifies an order has been cancelled
        """
        self.logger().info(f"The order {event.order_id} has been cancelled")

    def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
        """
        Method called when the connector notifies a buy order has been completed (fully filled)
        """
        self.logger().info(f"The buy order {event.order_id} has been completed")

    def did_complete_sell_order(self, event: SellOrderCompletedEvent):
        """
        Method called when the connector notifies a sell order has been completed (fully filled)
        """
        self.logger().info(f"The sell order {event.order_id} has been completed")

    def format_status(self) -> str:
        if not self.ready_to_trade:
            return "Market connectors are not ready."
        lines = []
        if self.all_candles_ready:
            lines.extend([
                "\n############################################ Market Data ############################################\n"])
            lines.extend([f"Exchange: {self.connector_name}\n"])

            lines.extend([f"Current Trend: {self.last_trend}"])
            lines.extend([f"Candles: {self.market_pair} | Interval: {self.trend_interval}"])
            lines.extend(["" + line for line in self.trend_candles_df.tail(3).to_string(index=False).split("\n")])
            lines.extend([
                "\n----------------------------------------------------------------------------------------------------\n"])

            lines.extend([f"Predict ATR: {self.atr_prediction}"])
            lines.extend([f"Candles: {self.market_pair} | Interval: {self.analysis_interval}"])
            lines.extend(["" + line for line in self.analysis_candles_df.tail(3).to_string(index=False).split("\n")])
            lines.extend([
                "\n----------------------------------------------------------------------------------------------------\n"])

            balance_df = self.get_balance_df()
            balance_df.drop(axis=1, columns={'Exchange'}, inplace=True)
            lines.extend(["" + line for line in balance_df.to_string(index=False).split("\n")])
            lines.extend([
                "\n----------------------------------------------------------------------------------------------------\n"])
            real_time_candles = self.analysis_candles.candles_df.tail(1).copy()
            real_time_candles.drop(axis=1,
                                   columns={
                                       'quote_asset_volume',
                                       'n_trades',
                                       'taker_buy_base_volume',
                                       'taker_buy_quote_volume'},
                                   inplace=True)
            real_time_candles["timestamp"] = pd.to_datetime(real_time_candles["timestamp"], unit="ms")
            lines.extend([f"Real time candle: "])
            lines.extend(["" + line for line in real_time_candles.to_string(index=False).split("\n")])

            lines.extend(["", "Active Orders:"])
            try:
                active_order = self.active_orders_df().drop('Exchange', axis=1)
                active_order['Price_diff %'] = (
                    abs(active_order['Price'] - float(self.current_price) / float(self.current_price)))
                lines.extend(["" + line for line in active_order.to_string(index=False).split("\n")])
            except ValueError:
                lines.extend(["", "No active maker orders."])

        else:
            lines.extend(["", "No data collected."])

        return "\n".join(lines)
