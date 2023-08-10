import React, { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { render, Drawable } from "@tensorflow/tfjs-vis";
import axios from "axios";

// 여기에 Quandl에서 생성한 API 키를 붙여 넣으십시오.
const API_KEY = "hgvbiFsBKpeS8aaxWxSx";
const numPredictionDays = 14; // 예측 일수를 변경하려면 이 값을 원하는 일수로 변경하세요.

const BTCPricePredictor = () => {
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

  // 간단한 RNN 모델 생성
  // const createModel = () => {
  //   const model = tf.sequential();

  //   model.add(
  //     tf.layers.lstm({
  //       units: 128,
  //       inputShape: [null, 1],
  //       kernelInitializer: "glorotUniform",
  //       recurrentInitializer: "glorotUniform"
  //     })
  //   );
  //   model.add(tf.layers.dropout({ rate: 0.2 }));
  //   model.add(
  //     tf.layers.dense({ units: 1, kernelInitializer: "glorotUniform" })
  //   );

  //   model.compile({
  //     optimizer: tf.train.adam(0.001),
  //     loss: tf.losses.meanSquaredError
  //   });

  //   return model;
  // };

  //gru 순환 신경망
  const createModel = () => {
    const model = tf.sequential();

    model.add(
      tf.layers.gru({
        units: 128,
        inputShape: [null, 1],
        kernelInitializer: "glorotUniform",
        recurrentInitializer: "glorotUniform"
      })
    );
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(
      tf.layers.dense({ units: 1, kernelInitializer: "glorotUniform" })
    );

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: tf.losses.meanSquaredError
    });

    return model;
  };

  //정규화
  const normalizeData = (data: number[]) => {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const normalizedData = data.map((value) => (value - min) / (max - min));
    return { normalizedData, min, max };
  };

  const trainModel = async (model: tf.LayersModel, data: number[]) => {
    // 데이터 정규화
    const { normalizedData, min, max } = normalizeData(data);

    // 마지막 7개 데이터를 예측에 사용하고 나머지를 훈련에 사용합니다.
    const trainData = normalizedData.slice(0, -7).map((price: any) => [price]);
    const testData = tf.tensor(
      normalizedData.slice(-7).map((price: any) => [price])
    );

    const X = tf.tensor(trainData.slice(0, -1)).reshape([-1, 1, 1]);
    const y = tf.tensor(trainData.slice(1)).reshape([-1, 1]);

    console.log(trainData, testData, X, y);

    await model.fit(X, y, {
      batchSize: 32,
      epochs: 20,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          console.log(`Epoch ${epoch}: loss = ${logs!.loss}`);
        }
      }
    });

    return { model, testData, min, max };
  };

  //정규화 역변환
  const denormalizeData = (
    normalizedValue: number,
    min: number,
    max: number
  ) => {
    return normalizedValue * (max - min) + min;
  };

  // 예측 함수
  const predict = (model: any, testData: any) => {
    // 모델을 사용하여 예측합니다.
    // 예측을 시작할 때 텐서를 3차원으로 재구성합니다.
    const input = testData.slice(0, -1).reshape([-1, 1, 1]);
    const predictionsTensor = model.predict(input); // 텐서 반환하기

    return predictionsTensor;
  };

  // 그래프를 표시하기 위한 간단한 플롯 함수
  const plot = async (
    predictions: any,
    trueValues: any,
    min: number,
    max: number
  ) => {
    const trueValuesArray = trueValues
      .arraySync()
      .map((value: any) => denormalizeData(value[0], min, max));
    const predictionsArray = predictions.arraySync();

    const trueValuesData = [...Array(7).keys()].map((x, i) => ({
      x: x,
      y: trueValuesArray[i]
    }));

    const predictionsData = [...Array(14).keys()].map((x, i) => ({
      x: x + 7,
      y: predictionsArray[i]
    }));
    console.log(trueValuesData, predictionsData);

    const data = {
      values: [trueValuesData, predictionsData],
      series: ["True values", "Predictions"]
    };

    const options = {
      zoomToFit: true,
      xLabel: "Time",
      yLabel: "BTC price"
    };

    const element = document.getElementById("plot");
    if (element) {
      await render.linechart(element as Drawable, data, options);
    } else {
      console.error('Element with id "plot" not found!');
    }
  };

  const run = async () => {
    console.log("Run: Model creation");
    const model = createModel();

    console.log("Run: Model training");
    const {
      model: trainedModel,
      testData,
      min,
      max
    } = await trainModel(model, data);

    console.log("Run: Model prediction");
    const predictionsTensor = predict(trainedModel, testData);

    // 결과 텐서를 배열로 변환하고 역변환을 적용합니다.
    const predictionsArray = predictionsTensor
      .arraySync()
      .map((value: any) => denormalizeData(value[0], min, max));

    // 역변환된 예측 값을 텐서로 변환합니다.
    const predictions = tf.tensor(
      predictionsArray.map((price: any) => [price])
    );

    console.log("Run: Plotting predictions");
    plot(predictions, testData, min, max);
  };

  return (
    <div>
      {isLoading ? (
        <p>데이터를 불러오는 중입니다...</p>
      ) : (
        <button onClick={run}>일주일 후 BTC 가격 예측 시작하기</button>
      )}
      <div id="plot"></div>
    </div>
  );
};

export default BTCPricePredictor;
