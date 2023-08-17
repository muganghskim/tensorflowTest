/*
RNN 모델 및 순환신경망 모델 적용 하였으나 유의미한 결과를 보여주지 못해 
금융 예측은 매우 어려워 보임 좀 더 실습하면서 데이터 변화를 지켜보며 새로운 모델을
적용 시키는 것이 좋은 방법일 것 같음
*/

import React, { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { render, Drawable } from "@tensorflow/tfjs-vis";
import axios from "axios";

// Quandl api는 데이터를 html 구조로 반환하여 사용하기가 까다로워 코인게코의 api를 사용함
// const API_KEY = "";
const numPredictionDays = 7; // 예측 일수를 변경하려면 이 값을 원하는 일수로 변경하세요.

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
    /* 드롭아웃 비율 조절: 과적합을 피하려면 dropout 레이어의 패러미터인 rate 값을 변경해 보세요. 
    일반적으로 0.2 ~ 0.5 사이의 값을 사용합니다. 이 값이 클수록 더 많은 뉴런이 드롭아웃되어 일반화 능력이 향상될 수 있습니다.
    */
    model.add(
      tf.layers.dense({ units: 1, kernelInitializer: "glorotUniform" })
    );

    // 배치 크기 및 에포크 변경: 학습 중 사용되는 배치 크기와 에포크를 변경하여 모델의 학습 방식을 수정할 수 있습니다.
    // 일반적으로 배치 크기를 늘리면 학습속도가 향상되고, 에포크를 늘리면 모델 성능이 향상될 수 있습니다. 그러나 과적합이 발생할 수도 있으므로 적절한 값을 맞추는게 중요합니다.
    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: tf.losses.meanSquaredError
    });

    return model;
  };

  // model.add(                           // 레이어를 추가하여 신경망의 깊이를 늘리고 더 복잡한 패턴 학습
  //   tf.layers.gru({
  //     units: 128,                      // units수는 뉴런에 해당함으로 해당값을 조정하여 복잡성 및 학습능력 변경
  //     inputShape: [null, 1],
  //     returnSequences: true,           // 이전 GRU 레이어의 출력을 다음 레이어의 입력으로 전달
  //     kernelInitializer: "glorotUniform",
  //     recurrentInitializer: "glorotUniform"
  //   })
  // );
  // model.compile({
  //   optimizer: tf.train.adam(0.01), // 0.001에서 0.01로 학습률 증가
  //   loss: tf.losses.meanSquaredError //
  // });

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

    // 마지막 400개 데이터를 예측에 사용하고 나머지를 훈련에 사용합니다.
    const trainData = normalizedData
      .slice(0, -400)
      .map((price: any) => [price]);
    const testData = tf.tensor(
      normalizedData.slice(-400).map((price: any) => [price])
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

  /*
데이터 전처리 및 분할에 대한 비율은 데이터의 특성, 문제의 종류, 모델의 복잡성 등에 따라 크게 달라질 수 있습니다. 따라서, 고정된 비율을 제공하기는 어렵습니다만, 
일반적인 접근 방식 및 가이드라인을 설명해 드리겠습니다. 먼저, 주어진 시계열 데이터셋을 훈련 데이터와 검증 데이터로 분할할 때 흔히 사용되는 비율은 80:20 또는 70:30입니다. 
학습에 사용될 최근 데이터의 기간(비율)을 정하는 것은 여러 가지 요소를 고려해야 합니다.

비즈니스 요구사항: 예측 범위에 따라 사용할 데이터를 선택해야 합니다. 예를 들어, 주간 영업을 예측해야 하는 경우, 지난 몇 주의 데이터를 사용하는 것이 이치에 맞습니다.

계절성 및 추세: 데이터의 일정 기간 동안 변화하는 계절성과 추세를 고려해야 합니다. 때에 따라 전체 데이터셋을 사용해야 하고, 경우에 따라 최근 몇 년, 몇 달 또는 
몇 주의 데이터만 사용하는 것이 좋을 수 있습니다.

노이즈 및 이상치: 전처리에서 이상치 및 노이즈 제거가 중요합니다. 이상치는 예측 성능을 저하시킬 수 있으므로, 필요한 경우 이상치 감지 및 제거 작업을 수행하세요.
비율을 조정하는 것 외에도 시계열 데이터에 대한 전처리 방법들은 다음과 같습니다:

차분(Differencing): 비정상 시계열 데이터를 정상 시계열로 변환하기 위해 이전 시계열 값과의 차이를 연산합니다.

이동 평균법(Moving Average): 데이터에서 잡음을 제거하기 위해 이동 평균을 산출하는 방법을 사용할 수 있습니다.

로그 변환(Log transformation): 로그를 취해 원래 값의 차이를 압축하는 것은 데이터의 변동성을 줄이는 데 도움이 됩니다.

시계열 분해(Time series decomposition): 시계열 데이터를 추세, 계절성, 잔차 성분으로 분해하여 별도로 처리한 후 다시 결합합니다.

따라서 관찰하고자 하는 패턴과 목표치에 따라 사용할 데이터의 길이를 알맞게 설정하고, 필요한 전처리를 적용하여 예측 성능을 향상시키는 것이 중요합니다. 
기본적으로 다양한 시나리오를 실험하고 어떤 목표치로 현재 설정된 파라미터를 수정하는 것이 가능하며, 이를 통해 최적의 예측 성능을 달성할 수 있습니다.
*/

  //정규화 역변환
  const denormalizeData = (
    normalizedValue: number,
    min: number,
    max: number
  ) => {
    return normalizedValue * (max - min) + min;
  };

  // 예측 함수
  const predict = (
    model: tf.LayersModel,
    testData: tf.Tensor,
    forecastDays: number,
    min: number,
    max: number
  ) => {
    const predictionsArray: number[] = [];

    // 기존 데이터 샘플의 마지막을 시작점으로 설정합니다.
    const startingIndex = testData.shape[0] - 1;
    console.log("첫데이터", startingIndex);

    // 첫 번째 예측을 위한 초기 입력 설정
    let input = testData.slice([startingIndex, 0], 1).reshape([-1, 1, 1]);

    for (let i = 0; i < forecastDays; i++) {
      // 현재 입력을 사용하여 예측
      const predictionsTensor = model.predict(input) as tf.Tensor;

      // 예측 값을 저장
      const currentPrediction: number = (
        predictionsTensor.arraySync() as number[][]
      )[0][0];
      predictionsArray.push(currentPrediction);

      // 들어온 새로운 값으로 입력을 업데이트 합니다.
      input = tf.tensor([[[currentPrediction]]]);
    }

    // 예측 결과를 역변환하여 반환합니다.
    return tf.tensor(
      predictionsArray.map((value) => denormalizeData(value, min, max))
    );
  };

  // todo: 하나씩 예측 모델 추가
  // const predictOneStep = (
  //   model: tf.LayersModel,
  //   inputData: any,
  //   min: any,
  //   max: any
  // ) => {
  //   const predictionsTensor = model.predict(inputData) as tf.Tensor;
  //   const currentPrediction = (
  //     predictionsTensor.arraySync() as number[][]
  //   )[0][0];
  //   return denormalizeData(currentPrediction, min, max);
  // };

  // const trainAndPredict = async (
  //   model: tf.LayersModel,
  //   trainData: any,
  //   min: any,
  //   max: any,
  //   forecastDays: any
  // ) => {
  //   const predictionsArray = [];

  //   for (let i = 0; i < forecastDays; i++) {
  //     // 모델 훈련
  //     await model.fit(trainData.inputs, trainData.outputs, {
  //       epochs: 1
  //       // 기타 훈련 옵션
  //     });

  //     // 가장 최근 데이터를 사용하여 입력을 생성합니다.
  //     const latestData = trainData.inputs.slice(
  //       trainData.inputs.shape[0] - 1,
  //       1
  //     );

  //     // 한 단계 미래 값 예측
  //     const prediction = predictOneStep(model, latestData, min, max);
  //     predictionsArray.push(prediction);

  //     // 예측값을 훈련 데이터(inputs, outputs)에 추가합니다.
  //     trainData.inputs = trainData.inputs.concat(
  //       tf.tensor([[[normalizeData(predictionsArray)]]])
  //     );
  //     trainData.outputs = trainData.outputs.concat(
  //       tf.tensor([normalizeData(predictionsArray)])
  //     );
  //   }

  //   return predictionsArray;
  // };

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

    const trueValuesData = trueValuesArray.map((value: number, i: number) => ({
      x: i,
      y: value
    }));

    const predictionsData = predictionsArray.map(
      (value: number, i: number) => ({
        x: trueValuesArray.length - 7 + i, // 수정됨: trueValuesData와 연결되도록 x 계산 변경
        y: value
      })
    );

    const combinedData = trueValuesData.slice(-7).concat(predictionsData); // 수정됨: 마지막 7개의 trueValuesData와 predictionsData를 연결

    const data = {
      values: [combinedData], // 수정됨: combinedData를 사용함
      series: ["Combined true values and predictions"] // 수정됨: 하나의 시리즈로만 표시
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

    // Forecast
    const forecastDays = 7;
    console.log("Run: Model prediction");
    //todo: 하나씩
    const predictionsTensor = predict(model, testData, forecastDays, min, max);
    // const predictionsArray = await trainAndPredict(
    //   model,
    //   trainData,
    //   min,
    //   max,
    //   7
    // );

    console.log("Run: Plotting predictions");
    plot(predictionsTensor, testData, min, max);
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
