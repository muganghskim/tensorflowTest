import React, { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { render, Drawable } from "@tensorflow/tfjs-vis";
import axios from "axios";

const BTCPredictorV2 = () => {
  const [data, setData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      const response = await axios.get(
        `https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=2000&interval=daily`
      );
      console.log(response.data);
      setData(processData(response.data));
      setIsLoading(false);
    }
    fetchData();
  }, []);

  const processData = (data: any) => {
    const prices = data.prices.map((entry: any) => Math.round(entry[1])); // 가격을 정수로 반올림
    console.log(prices);
    return prices;
  };

  const run = async () => {};

  return (
    <div>
      {isLoading ? (
        <p>데이터를 불러오는 중입니다...</p>
      ) : (
        <button onClick={run}>1일 후 BTC 가격 예측 시작하기</button>
      )}
      <div id="plot"></div>
    </div>
  );
};

export default BTCPredictorV2;
