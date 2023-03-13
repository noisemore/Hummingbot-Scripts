import logging
from decimal import Decimal
from typing import Dict
import time
from typing import List

import pandas as pd
import pandas_ta as ta  # noqa: F401
from hummingbot.connector.connector_base import ConnectorBase
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from hummingbot.connector.utils import split_hb_trading_pair
from hummingbot.core.data_type.common import TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
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
    data_interval = int(14400)  # 默認時間間隔為 4 小時，以秒為單位表示
    if str("s") in _data_interval:  # 如果時間間隔包含 "s"，表示以秒為單位
        data_interval = _data_interval.replace(str("s"), str(""))  # 去除 "s" 字符
        data_interval = int(data_interval)  # 將字符串轉換為整數
    if str("m") in _data_interval:  # 如果時間間隔包含 "m"，表示以分鐘為單位
        data_interval = _data_interval.replace(str("m"), str(""))  # 去除 "m" 字符
        data_interval = int(data_interval)  # 將字符串轉換為整數
        data_interval = int(data_interval * 60)  # 將分鐘轉換為秒
    elif "h" in _data_interval:  # 如果時間間隔包含 "h"，表示以小時為單位
        data_interval = _data_interval.replace(str("h"), str(""))  # 去除 "h" 字符
        data_interval = int(data_interval)  # 將字符串轉換為整數
        data_interval = int(data_interval * 60 * 60)  # 將小時轉換為秒
    elif "d" in _data_interval:  # 如果時間間隔包含 "d"，表示以天為單位
        data_interval = _data_interval.replace(str("d"), str(""))  # 去除 "d" 字符
        data_interval = int(data_interval)  # 將字符串轉換為整數
        data_interval = int(data_interval * 60 * 60 * 24)  # 將天數轉換為秒
    elif "w" in _data_interval:  # 如果時間間隔包含 "w"，表示以週為單位
        data_interval = _data_interval.replace(str("w"), str(""))  # 去除 "w" 字符
        data_interval = int(data_interval)  # 將字符串轉換為整數
        data_interval = int(data_interval * 60 * 60 * 24 * 7)  # 將週數轉換為秒
    return data_interval  # 返回轉換後的時間間隔，以秒為單位表示


class spot_price_prediction_market_making(ScriptStrategyBase):
    connector_name = str("binance")  # 交易所名稱
    data_connector = str("binance")  # 數據連接器名稱
    market_pair = str("RLC-USDT")  # 市場交易對
    coin_base, coin_quote = split_hb_trading_pair(market_pair)  # 將交易對拆分為基礎幣和報價幣
    markets = {connector_name: {market_pair}}  # 交易所市場信息

    # Trend Parameters
    trend_interval = str("4h")  # 趨勢分析的時間間隔
    trend_candles = CandlesFactory.get_candle(connector=data_connector,
                                              trading_pair=market_pair,
                                              interval=trend_interval,
                                              max_records=60)  # 獲取趨勢分析所需的K線數據

    vtx_length = int(14)  # Vortex指標的計算長度

    st_length = int(6)  # SuperTrend指標的計算長度
    st_multiplier = float(2.618)  # SuperTrend指標的乘數

    adx_length = int(14)  # ADX指標的計算長度

    trend_candles_df = pd.DataFrame()  # 儲存趨勢分析的K線數據
    last_trend = bool(True)  # 上一個趨勢方向（True表示上漲，False表示下跌）
    action_signal = str("")  # 執行的操作信號（"Buy"表示買入，"Sell"表示賣出，""表示不進行任何操作）
    have_data = bool(False)  # 是否獲取到了足夠的K線數據

    analysis_interval = str("1m")  # 進行價格預測的時間間隔
    analysis_candles = CandlesFactory.get_candle(connector=data_connector,
                                                 trading_pair=market_pair,
                                                 interval=analysis_interval,
                                                 max_records=60)  # 獲取進行價格預測所需的K線數據

    mom_len = int(10)  # 價格預測中的動量指標計算長度
    mfi_len = int(14)  # 價格預測中的貨幣流量指標計算長度
    atr_len = int(14)  # 價格預測中的平均真實範圍
    atr_prediction = Decimal("0.00")  # 預測的平均真實範圍
    analysis_candles_df = pd.DataFrame()  # 儲存進行價格預測的K線數據

    build_interval = str("30s")  # 建立庫存的時間間隔

    mid_price = Decimal("0.00")  # 市場的中間價格
    total_budget = Decimal("100")  # 總預算
    order_usd = Decimal("50")  # 每筆訂單的美元金額
    min_order_size = Decimal("10.0")  # 最小交易量
    stop_loss = Decimal('0.1')  # 止損閾值

    u_buy_atr_multiplier = Decimal("2")  # 上漲時買入的ATR倍數
    u_sell_atr_multiplier = Decimal("2.5")  # 上漲時賣出的ATR倍數
    d_buy_atr_multiplier = Decimal("2.5")  # 下跌時買入的ATR倍數
    d_sell_atr_multiplier = Decimal("2")  # 下跌時賣出的ATR倍數
    closest_order_percentage = Decimal("0.003")  # 最小訂單比例
    farthest_order_percentage = Decimal("0.015")  # 最大訂單比例

    last_ordered_ts = int(0)
    order_interval = int(5)

    run_the_bot = bool(True)  # 是否運行機器人策略

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

        if self.run_the_bot:
            if self.all_candles_ready:
                # 已蒐集到足夠的數據，可以開始進行策略運行
                # ========================================== Technical Analysis Start =========================================#

                # 獲取趨勢數據和 ATR 預測
                if not self.have_data:
                    self.trend_candles_df, self.last_trend, self.action_signal = self.get_trend_signal()
                    self.analysis_candles_df, self.atr_prediction = self.get_atr_prediction()
                    self.have_data = True
                else:
                    # 根據指定的時間間隔更新趨勢數據和 ATR 預測
                    if self.run_instantly(self.trend_interval):
                        self.trend_candles_df, self.last_trend, self.action_signal = self.get_trend_signal()
                    if self.run_instantly(self.analysis_interval):
                        self.analysis_candles_df, self.atr_prediction = self.get_atr_prediction()

                # ========================================== Technical Analysis End ===========================================#

                # /////////////////////////////////////////////////////////////////////////////////////////////////////////////#

                # ========================================== Order Management Start ===========================================#
                # 檢查是否有足夠庫存及資金
                base_checker, quote_checker = self.inventory_checker()

                # 檢查止損閾值
                if self.get_stop_loss():
                    if base_checker:
                        # 如果當前價格低於止損閾值，則賣出所有持倉並且停止策略
                        self.logger().info("Stop loss threshold reached, selling all inventory...")
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
                        # 如果沒有庫存或者庫存太小，停止策略但是要手動檢查
                        self.logger().info("No inventory or inventory is too small, no need to stop loss...")
                        self.cancel_all_orders()
                        self.run_the_bot = False
                else:
                    if not base_checker:
                        if len(self.get_active_orders(connector_name=self.connector_name)) == 0:
                            # 有下單就不要叫叫叫
                            self.logger().info("No inventory, checking if we can build inventory...")
                        if self.run_instantly(self.build_interval):
                            # 庫存價值小於 10, 檢查是否有足夠金額並下單
                            self.cancel_all_orders()
                            self.build_inventory()
                    elif not quote_checker:
                        if len(self.get_active_orders(connector_name=self.connector_name)) == 0:
                            # 有下單就不要叫叫叫
                            self.logger().info("No money, checking if we can sell inventory...")
                        if self.run_instantly(self.build_interval):
                            # 穩定幣價值小於 10, 檢查是否有足夠庫存並下單
                            self.cancel_all_orders()
                            self.get_money()
                    else:
                        # 中間價有變動就下單或者時間到就下單
                        if self.mid_price != self.current_price or self.last_ordered_ts < (self.current_timestamp - self.order_interval):
                            self.cancel_all_orders()

                            # 根據 ATR 和當前趨勢設置買入和賣出目標價格
                            buy_atr_multiplier = self.u_buy_atr_multiplier if self.last_trend else self.d_buy_atr_multiplier
                            sell_atr_multiplier = self.u_sell_atr_multiplier if self.last_trend else self.d_sell_atr_multiplier

                            # 计算最小和最大买入价格和卖出价格
                            closest_buy_price = self.current_price * (Decimal('1') - self.closest_order_percentage)
                            farthest_buy_price = self.current_price * (Decimal('1') - self.farthest_order_percentage)
                            closest_sell_price = self.current_price * (Decimal('1') + self.closest_order_percentage)
                            farthest_sell_price = self.current_price * (Decimal('1') + self.farthest_order_percentage)

                            # 根据 ATR 预测和乘数计算预测的买入和卖出目标价格
                            predict_buy_target_price = self.current_price - (self.atr_prediction * buy_atr_multiplier)
                            predict_sell_target_price = self.current_price + (self.atr_prediction * sell_atr_multiplier)

                            # 根据最小和最大买入价格和卖出价格设置买入和卖出目标价格
                            if predict_buy_target_price >= closest_buy_price:
                                buy_target_price = closest_buy_price
                            elif predict_buy_target_price <= farthest_buy_price:
                                buy_target_price = farthest_buy_price
                            else:
                                buy_target_price = predict_buy_target_price

                            if predict_sell_target_price < closest_sell_price:
                                sell_target_price = closest_sell_price
                            elif predict_sell_target_price > farthest_sell_price:
                                sell_target_price = farthest_sell_price
                            else:
                                sell_target_price = predict_sell_target_price

                            buy_order_amount = self.buy_amount(buy_target_price)
                            sell_order_amount = self.sell_amount(sell_target_price)

                            buy_order = OrderCandidate(trading_pair=self.market_pair, is_maker=True,
                                                       order_type=OrderType.LIMIT,
                                                       order_side=TradeType.BUY, amount=buy_order_amount,
                                                       price=buy_target_price)

                            sell_order = OrderCandidate(trading_pair=self.market_pair, is_maker=True,
                                                        order_type=OrderType.LIMIT,
                                                        order_side=TradeType.SELL, amount=sell_order_amount,
                                                        price=sell_target_price)

                            orders = [buy_order, sell_order]

                            proposal: List[OrderCandidate] = orders
                            proposal_adjusted: List[OrderCandidate] = self.adjust_proposal_to_budget(proposal)
                            self.place_orders(proposal_adjusted)

                            self.mid_price = self.current_price
                            self.last_ordered_ts = self.current_timestamp
            else:
                # 如果沒有數據，則等待
                self.logger().info("Waiting for data...")
        else:
            # 如果不運行策略，則等待
            self.logger().info("Bot is not running...")

            # ========================================== Order Management End =============================================#

    def run_instantly(self, input_interval):
        """
        Run the bot instantly
        """
        return self.current_timestamp == ((self.current_timestamp // interval_in_sec(input_interval)) * interval_in_sec(input_interval))

    def get_trend_signal(self):
        # 獲取趨勢蠟燭圖數據
        trend_candles_df = self.trend_candles.candles_df
        # 刪除不需要的列
        trend_candles_df.drop(axis=1, columns={'quote_asset_volume', 'n_trades', 'taker_buy_base_volume',
                                               'taker_buy_quote_volume'}, inplace=True)

        # 計算 VTX 指標
        vtx_props = f"_{self.vtx_length}"
        trend_candles_df.ta.vortex(append=True)
        trend_candles_df = trend_candles_df.rename({f'VTXP{vtx_props}': 'VTXP', f'VTXM{vtx_props}': 'VTXM'},
                                                   axis='columns')
        trend_candles_df['VTX_uptrend'] = trend_candles_df.apply(lambda row: row['VTXP'] > row['VTXM'],
                                                                 axis=1).astype(bool)

        # 計算 SuperTrend 指標
        st_props = f"_{self.st_length}_{self.st_multiplier}"
        trend_candles_df.ta.supertrend(length=self.st_length, multiplier=self.st_multiplier, append=True)
        trend_candles_df = trend_candles_df.rename({f'SUPERTd{st_props}': 'ST_uptrend'}, axis='columns')
        trend_candles_df['ST_uptrend'] = trend_candles_df['ST_uptrend'].replace({1: True, -1: False}).astype(bool)

        # 計算 ADX 指標
        adx_props = f"_{self.adx_length}"
        trend_candles_df.ta.adx(length=self.adx_length, append=True)
        trend_candles_df = trend_candles_df.rename(
            {f'ADX{adx_props}': 'ADX', f'DMP_{self.adx_length}': 'DMP', f'DMN_{self.adx_length}': 'DMN'},
            axis='columns')
        trend_candles_df['ADX_uptrend'] = trend_candles_df.apply(lambda row: row['DMP'] > row['DMN'],
                                                                 axis=1).astype(bool)

        # 判斷總趨勢
        trend_candles_df['uptrend'] = (trend_candles_df['VTX_uptrend'].astype(int) +
                                       trend_candles_df['ST_uptrend'].astype(int) +
                                       trend_candles_df['ADX_uptrend'].astype(int)) >= 2

        # 刪除不需要的列
        trend_candles_df.drop(axis=1, columns={'VTXP', 'VTXM', f'SUPERT{st_props}', f'SUPERTl{st_props}',
                                               f'SUPERTs{st_props}', 'ADX', 'DMP', 'DMN', 'VTX_uptrend',
                                               'ST_uptrend', 'ADX_uptrend'}, inplace=True)
        # 將時間戳轉換為 datetime 格式
        trend_candles_df["timestamp"] = pd.to_datetime(trend_candles_df["timestamp"], unit="ms")

        # 判斷趨勢信號
        pre_trend = bool(trend_candles_df.iloc[-2]['uptrend'])  # 上一次的趨勢
        last_trend = bool(trend_candles_df.iloc[-1]['uptrend'])  # 最後一次的趨勢

        # 根據趨勢信號生成動作信號
        if last_trend is True and pre_trend is False:
            action_signal = str("Long")
        elif last_trend is False and pre_trend is True:
            action_signal = str("Short")
        else:
            action_signal = str("No Action")

        # 返回趨勢蠟燭圖數據、最後一次趨勢、動作信號
        return trend_candles_df, last_trend, action_signal

    def get_atr_prediction(self):
        # 獲取分析蠟燭圖數據
        analysis_candles_df = self.analysis_candles.candles_df.copy()
        # 刪除不需要的列
        analysis_candles_df.drop(axis=1,
                                 columns={
                                     'quote_asset_volume',
                                     'n_trades',
                                     'taker_buy_base_volume',
                                     'taker_buy_quote_volume'},
                                 inplace=True)
        # 計算 Trading Volume、HLC3 和 True Range 指標
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

        # 將 High 和 Low 列中的數據向前位移一位
        analysis_candles_df['atr_1'] = analysis_candles_df['atr'].shift(1)
        analysis_candles_df.dropna(inplace=True)

        # 定義特徵和目標變量
        features = analysis_candles_df[['volume', 'high', 'low', 'close', 'tr', 'mom', 'mfi']]
        target = analysis_candles_df['atr_1']

        # 分割訓練集和測試集
        features_train, features_test, target_train, target_test = train_test_split(features, target,
                                                                                    test_size=0.35, random_state=0)

        # 創建線性回歸模型並訓練它
        lr = LinearRegression()
        lr.fit(features_train, target_train)

        # 在測試集上進行預測
        target_pred = lr.predict(features_test)

        # 計算均方根誤差（RMSE）
        # mse = mean_squared_error(targets_test, targets_pred)
        # rmse = mse ** 0.5

        # 對特徵進行預測
        features = features.dropna()
        analysis_candles_df['target_pred'] = lr.predict(features)
        # 將時間戳轉換為 datetime 格式
        analysis_candles_df["timestamp"] = pd.to_datetime(analysis_candles_df["timestamp"], unit="ms")
        atr_pred = Decimal(str(analysis_candles_df['target_pred'].iloc[-1]))
        # 刪除不需要的列
        analysis_candles_df.drop(axis=1, columns={'target_pred', 'tr', 'mom', 'mfi', 'atr_1'}, inplace=True)

        # 返回分析蠟燭圖數據和 ATR
        return analysis_candles_df, atr_pred

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

    def inventory_checker(self):
        current_value = self.connectors[self.connector_name].get_balance(self.coin_base) * self.current_price
        # 檢查當前賬戶中的 base價值是否大於 10
        base_checker = bool(current_value > Decimal("10"))
        # 檢查當前賬戶中的 quote價值是否大於 10
        quote_checker = bool(self.connectors[self.connector_name].get_balance(self.coin_quote) > Decimal("10"))
        return base_checker, quote_checker

    def get_stop_loss(self):
        # 計算當前賬戶總價值
        current_value = self.connectors[self.connector_name].get_balance(self.coin_quote) + self.connectors[
            self.connector_name].get_balance(self.coin_base) * self.current_price
        # 計算停損價值
        stop_loss_value = self.total_budget * (Decimal('1') - self.stop_loss)

        # 如果當前價值小於停損價值，則返回 True，否則返回 False
        return bool(current_value < stop_loss_value)

    def build_inventory(self):
        current_value = self.connectors[self.connector_name].get_balance(self.coin_base) * self.current_price
        usd_balance = self.connectors[self.connector_name].get_balance(self.coin_quote)
        if usd_balance >= Decimal(self.order_usd):
            # 如果當前庫存價值小於 order_usd，則補足庫存到 order_usd
            self.buy(
                connector_name=self.connector_name,
                trading_pair=self.market_pair,
                amount=(Decimal(self.order_usd) - current_value) / self.connectors[self.connector_name].get_price(
                    self.market_pair, True),
                order_type=OrderType.LIMIT,
                price=self.connectors[self.connector_name].get_price(
                    self.market_pair, True)
            )
        else:
            # 穩定幣不足，不進行買入
            self.logger().info(f"Insufficient {self.coin_quote} balance to buy {self.coin_base}")

    def get_money(self):
        current_value = self.connectors[self.connector_name].get_balance(self.coin_base) * self.current_price
        buy_atr_multiplier = self.u_buy_atr_multiplier if self.last_trend else self.d_buy_atr_multiplier
        if current_value >= Decimal(self.order_usd):
            # 如果當前庫存價值大於 order_usd，則賣出庫存到 order_usd
            self.sell(
                connector_name=self.connector_name,
                trading_pair=self.market_pair,
                amount=(current_value - Decimal(self.order_usd)) / self.connectors[self.connector_name].get_price(
                    self.market_pair, False),
                order_type=OrderType.LIMIT,
                price=self.connectors[self.connector_name].get_price(self.market_pair, False)
            )
        else:
            # 庫存不足，不進行賣出
            self.logger().info(f"Insufficient {self.coin_base} balance to sell {self.coin_quote}")

    def buy_amount(self, buy_target_price):
        quote_balance = self.connectors[self.connector_name].get_balance(self.coin_quote)
        base_balance = self.connectors[self.connector_name].get_balance(self.coin_base)
        # 如果 USDT 餘額大於等於設置的下單金額 'order_usd'，則設置買入數量為 'order_usd / buy_target_price'
        if quote_balance >= Decimal(self.order_usd):
            buy_amount = self.order_usd / buy_target_price
        # 如果 USDT 餘額小於 'order_usd'，但大於等於 'min_order_size'，則設置買入數量為 'connectors[self.connector_name].get_balance(self.coin_quote) / buy_target_price'
        elif self.min_order_size <= quote_balance < Decimal(self.order_usd):
            buy_amount = (Decimal(self.order_usd) - (base_balance * self.current_price)) / buy_target_price
        # 如果 USDT 餘額小於 'min_order_size'，則設置買入數量為 0，不進行買入
        else:
            buy_amount = Decimal('0.0')

        # 返回買入數量
        return buy_amount

    def sell_amount(self, sell_target_price):
        base_balance = self.connectors[self.connector_name].get_balance(self.coin_base)
        current_value = base_balance * self.current_price
        # 如果庫存大小（base coin 數量 * 當前價格）大於等於 'min_order_size'，並且小於等於 'order_usd'，則設置賣出數量為庫存大小（base coin 數量）
        if Decimal(self.min_order_size) < current_value <= Decimal(self.order_usd):
            sell_order_amount = base_balance
        # 如果庫存價值大於 'order_usd'，則設置賣出數量為 'order_usd / 預測價格'
        elif Decimal(self.order_usd) < current_value:
            sell_order_amount = self.order_usd / sell_target_price
        # 如果庫存大小小於 'min_order_size'，則不進行賣出
        else:
            sell_order_amount = Decimal('0.0')

        # 返回賣出數量
        return sell_order_amount

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        proposal_adjusted = self.connectors[self.connector_name].budget_checker.adjust_candidates(proposal, all_or_none=True)
        return proposal_adjusted

    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        for order in proposal:
            self.place_order(connector_name=self.connector_name, order=order)

    def place_order(self, connector_name: str, order: OrderCandidate):
        if order.order_side == TradeType.SELL:
            self.sell(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                      order_type=order.order_type, price=order.price)
        elif order.order_side == TradeType.BUY:
            self.buy(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                     order_type=order.order_type, price=order.price)

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
                active_order['Price_diff %'] = abs(active_order['Price'] - float(self.current_price)) / float(self.current_price) * float(100)
                lines.extend(["" + line for line in active_order.to_string(index=False).split("\n")])
            except ValueError:
                lines.extend(["", "No active maker orders."])

        else:
            lines.extend(["", "No data collected."])

        return "\n".join(lines)
