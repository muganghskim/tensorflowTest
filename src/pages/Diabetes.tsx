/*
상태 설정:
model: 훈련된 TensorFlow 모델을 저장합니다. 초기값은 null입니다.
trainingStatus: 모델 훈련 상태를 나타내는 문자열입니다.
predictButtonDisabled: 예측 버튼 활성화 여부를 결정하는 불린 값입니다.
predictionResult: 예측 결과를 사용자에게 표시할 문자열입니다.

getData(): 당뇨병 데이터셋을 CSV 형식으로 가져와 특징 이름과 데이터를 반환합니다.

createModel(inputShape: any): 주어진 입력 모양을 가지는 Dense 층을 포함하는 TensorFlow.js 모델을 생성합니다.

trainModel(model: any, data: any): 모델을 주어진 데이터로 훈련하고 훈련 이력을 반환합니다. 훈련 중에 배치 크기, 에포크, 셔플링 및 검증 분할 사용을 설정합니다.

run(): 데이터를 가져오고 모델을 생성하고 훈련하는 비동기 함수입니다. 훈련 시작, 진행 및 완료 시 상태 업데이트를 수행합니다.

handlePrediction(event: React.FormEvent<HTMLFormElement>): 사용자가 입력한 데이터를 가져와 모델을 사용하여 예측하고 결과를 화면에 표시하는 이벤트 핸들러입니다.

리턴된 JSX는 다음 요소를 포함합니다:
훈련 및 예측에 관련된 제목과 상태 정보 표시
훈련 모델 버튼 (run 함수 실행)
사용자 입력 폼과 예측 버튼 (handlePrediction 함수 실행)
예측 결과 표시
*/

import React, { useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "../../src/assets/scss/diabetes.css";

// Add type definition for the data values.
type Data = {
  featureNames: string[];
  data: number[][];
};

const Diabetes: React.FC = () => {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [trainingStatus, setTrainingStatus] =
    useState<string>("Not trained yet.");
  const [predictButtonDisabled, setPredictButtonDisabled] =
    useState<boolean>(true);
  const [predictionResult, setPredictionResult] = useState<string>("");

  const getData = async (): Promise<Data> => {
    // 데이터셋 로드
    const req = await fetch(
      "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    );
    const csvText = await req.text();
    const csvLines = csvText.split("\n");

    // 특징 이름과 데이터 추출
    const featureNames = csvLines[0].split(",");
    const data = csvLines
      .slice(1)
      .map((line) => line.split(",").map((x) => parseFloat(x)));

    return { featureNames, data };
  };

  function createModel(inputShape: any) {
    const model = tf.sequential();

    model.add(
      tf.layers.dense({ units: 16, activation: "relu", inputShape: inputShape })
    );
    model.add(tf.layers.dense({ units: 8, activation: "relu" }));
    model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: "binaryCrossentropy",
      metrics: ["accuracy"]
    });
    return model;
  }

  async function trainModel(model: any, data: any) {
    const batchSize = 32;
    const epochs = 50;

    const xVals = data.map((d: any) => d.slice(0, -1));
    const yVals = data.map((d: any) => parseInt(d[d.length - 1]));

    const size = xVals.length;

    const inputTensor = tf.tensor2d(xVals, [size, xVals[0].length]);
    const labelTensor = tf.tensor2d(yVals, [size, 1]);

    const history = await model.fit(inputTensor, labelTensor, {
      batchSize,
      epochs,
      shuffle: true,
      validationSplit: 0.1
    });

    return history;
  }

  const run = async () => {
    const { featureNames, data } = await getData();
    const model = createModel([featureNames.length - 1]);
    setModel(model);

    setTrainingStatus("Training in progress...");
    setPredictButtonDisabled(true);

    const history = await trainModel(model, data);

    setTrainingStatus("Model trained.");
    setPredictButtonDisabled(false);
  };

  const handlePrediction = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!model) {
      console.error("Model is null, cannot make prediction");
      return;
    }

    const form = event.currentTarget;

    // 입력값을 가져옵니다.
    const inputData = {
      pregnancies: (form.elements.namedItem("pregnancies") as HTMLInputElement)
        .valueAsNumber,
      glucose: (form.elements.namedItem("glucose") as HTMLInputElement)
        .valueAsNumber,
      bloodPressure: (
        form.elements.namedItem("blood-pressure") as HTMLInputElement
      ).valueAsNumber,
      skinThickness: (
        form.elements.namedItem("skin-thickness") as HTMLInputElement
      ).valueAsNumber,
      insulin: (form.elements.namedItem("insulin") as HTMLInputElement)
        .valueAsNumber,
      bmi: (form.elements.namedItem("bmi") as HTMLInputElement).valueAsNumber,
      diabetesPedigree: (
        form.elements.namedItem("diabetes-pedigree") as HTMLInputElement
      ).valueAsNumber,
      age: (form.elements.namedItem("age") as HTMLInputElement).valueAsNumber
    };

    const testInput = tf.tensor2d([Object.values(inputData)], [1, 8]);

    // const predictionData = model.predict(testInput) as tf.Tensor<tf.Rank>[];
    // const prediction =
    //   predictionData.length > 0 ? tf.concat(predictionData).dataSync()[0] : 0;
    const predictionData = model.predict(testInput) as tf.Tensor<tf.Rank>;
    const prediction = predictionData.dataSync()[0];
    const predictionText =
      prediction > 0.5
        ? "당뇨병 확률이 50% 이상으로 위험합니다."
        : "당뇨병 확률이 50% 이하입니다. 건강관리에 유념해 주세요.";
    setPredictionResult(
      `당뇨병 예측: ${predictionText} (${(prediction * 100).toFixed(2)}%)`
    );
  };

  return (
    <div className="diabet">
      <h1>당뇨병 예측 모델</h1>
      <div id="training-status">훈련 상태: {trainingStatus}</div>
      <button id="train-model" onClick={run}>
        훈련준비
      </button>
      <form id="user-input-form" onSubmit={handlePrediction}>
        <label htmlFor="pregnancies">임신횟수:</label>
        <input type="number" id="pregnancies" name="pregnancies" required />
        <br />

        <label htmlFor="glucose">공복혈당치(성인 평균):</label>
        <input
          type="number"
          id="glucose"
          name="glucose"
          required
          defaultValue="100"
        />
        <br />

        <label htmlFor="blood-pressure">혈압:</label>
        <input
          type="number"
          id="blood-pressure"
          name="blood-pressure"
          required
        />
        <br />

        <label htmlFor="skin-thickness">피하지방두께(성인 평균):</label>
        <input
          type="number"
          id="skin-thickness"
          name="skin-thickness"
          required
          defaultValue="20"
        />
        <br />

        <label htmlFor="insulin">인슐린수치(성인 평균):</label>
        <input
          type="number"
          id="insulin"
          name="insulin"
          required
          defaultValue="15"
        />
        <br />

        <label htmlFor="bmi">BMI:</label>
        <input type="number" id="bmi" name="bmi" step="0.01" required />
        <br />

        <label htmlFor="diabetes-pedigree">당뇨 가족력 수치(성인 평균):</label>
        <input
          type="number"
          id="diabetes-pedigree"
          name="diabetes-pedigree"
          step="0.001"
          required
          defaultValue="0.4"
        />
        <br />

        <label htmlFor="age">나이:</label>
        <input type="number" id="age" name="age" required />
        <br />
        <button type="submit" id="predict" disabled={predictButtonDisabled}>
          예측시작(예측전 훈련 준비를 끝내주세요.)
        </button>
      </form>
      <div id="result">{predictionResult}</div>
    </div>
  );
};

export default Diabetes;
