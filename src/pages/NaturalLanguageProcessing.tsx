/* 현재 주어진 코드와 데이터셋은 매우 간단하여 예측 결과가 이상하게 나올 수 있습니다. 유의미한 결과를 얻으려면 다음과 같이 설정을 개선해야 합니다:

데이터셋 확장: 현재 사용된 데이터셋은 매우 제한적입니다. 실제 자연어 처리에서는 대표적이고 큰 규모의 데이터셋을 사용하여 학습합니다. 
텍스트의 유형과 목적에 따라 다양한 데이터셋을 사용할 수 있습니다.

레이어 및 뉴런 조정: 모델의 구조를 보완하고, 필요한 레이어와 뉴런 수를 수정합니다. 
예를 들어, 시퀀스 데이터 처리에 적합한 LSTM 또는 GRU 레이어를 사용하는 것이 좋습니다.

학습 조절: 에포크 수를 늘리거나, 학습률을 조절하며 학습의 효과를 최적화 합니다. 
너무 작은 에포크는 부정확한 결과를, 너무 큰 에포크는 과적합을 초래할 수 있으므로 적절한 값을 찾아야 합니다.

텍스트 전처리 개선: 텍스트 전처리에 토큰화, 정규화, 불용어 제거 등의 과정을 적용하여 입력 데이터의 품질을 높입니다.

평가 및 검증: 테스트 데이터셋을 사용하여 모델의 성능을 평가하고, 과적합 여부를 검사합니다. 이를 통해 모델의 일반화 능력을 향상시킬 수 있습니다.
위와 같이 설정을 개선하면 예측 결과의 품질이 향상되어 유의미한 결과를 확인할 수 있을 것입니다. 
이를 위해 자연어 처리에 대한 추가적인 학습과 문헌 조사를 통해 최적화된 모델 구조와 데이터셋을 찾는 것이 좋습니다. 

자연어 처리 모델의 정밀도를 높이는 방법은 여러 가지가 있습니다. 복잡한 대화를 처리하기에 지금 구현하신 모델이 다소 단순합니다. 몇 가지 시도해볼 수 있는 방법을 제시해 드리겠습니다:

데이터셋 확장: 현재 데이터셋이 상대적으로 매우 작습니다. 더 큰 데이터셋으로 모델을 학습시킬 경우, 훨씬 더 다양한 대화 상황에 대응할 수 있게 됩니다.

모델 구조 개선: 여러 LSTM 레이어를 겹쳐서 더 복잡한 구조의 네트워크를 구성할 수 있습니다. 또한, 양방향 LSTM(BiLSTM) 사용하는 것도 고려해 볼 만한 방법입니다.

사전 학습된 모델 사용: GPT 같은 사전 학습된 모델을 사용하면 대화 처리 성능을 크게 향상시킬 수 있습니다. 
이러한 모델은 이미 수많은 텍스트 데이터로 사전 학습되어서, 언어 구조와 시맨틱 기반으로 자연스러운 회답 생성이 가능합니다.

학습 에포크와 배치 크기 조절: 현재 20 에포크 및 배치 크기 32로 설정되어 있습니다.
 경우에 따라서는 학습 에포크를 늘리거나, 배치 크기를 조절하는 것도 성능 향상에 도움이 될 수 있습니다. 
 이를 통해 모델이 데이터셋에 더 적응하게 되고, 정밀도가 향상되기를 기대할 수 있습니다.

생성 프로세스 개선: 현재 추론 과정에서 목표 텍스트를 생성하기 위해 코드 내에서 sample() 함수를 사용하고 있습니다. 
생성 프로세스를 개선하는 방식으로 Beam Search 등의 기법을 사용할 수 있습니다. 
Beam Search는 모델의 출력 결과를 버리지 않고 동시에 다양한 후보 생성 길을 탐색하면서 안정적인 출력을 만들어냅니다. 
이를 통해 생성된 문장의 품질이 향상된 대화 결과를 얻을 수 있습니다.

모델의 정밀도를 높일 수 있는 최선의 방법은 모델의 구조와 학습 데이터를 개선하여 보다 현실적인 대화 상황에 대응 할 수 있게 만드는 것입니다. 
계속해서 모델 구조와 학습 데이터셋을 실험하며 개선 시도를 할 경우 자연스러운 답변을 생성할 수 있는 자연어 처리 모델을 만드실 수 있습니다.
*/

import React, { useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { preprocessText, oneHotEncode, sample } from "./helpers";
import { message } from "./message";

const NaturalLanguageProcessing: React.FC = () => {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [text, setText] = useState<string>("");
  const [answer, setAnswer] = useState<string>("");

  const createModel = async () => {
    console.log("모델 생성 시작!");
    const model = tf.sequential();

    console.log("첫 번째 LSTM 층 추가 중...");
    model.add(
      tf.layers.lstm({
        units: 128,
        inputShape: [maxlengthInput, 26],
        returnSequences: false,
        kernelInitializer: "glorotNormal", // 초기화 변경
        recurrentInitializer: "glorotNormal", // 초기화 변경
        biasInitializer: "zeros"
      })
    );

    console.log("RepeatVector 층 추가 중...");
    model.add(tf.layers.repeatVector({ n: maxlengthOutput }));

    console.log("두 번째 LSTM 층 추가 중...");
    model.add(tf.layers.lstm({ units: 128, returnSequences: true }));

    console.log("TimeDistributed 층 추가 중...");
    model.add(
      tf.layers.timeDistributed({
        layer: tf.layers.dense({ units: 26, activation: "softmax" })
      })
    );

    console.log("모델 컴파일 중...");
    model.compile({
      optimizer: "adam",
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"]
    });

    console.log("모델 설정 완료!");
    setModel(model);
  };

  const maxlengthInput = 10;
  const maxlengthOutput = 10;

  const trainModel = async () => {
    const trainLogs = [];
    const batchSize = 32;
    const epochs = 2;
    if (!model) {
      return;
    }

    // 데이터셋 준비 (실제 프로젝트에서는 더 대표적인 데이터셋 사용)
    const dataFromApi = message;

    // 원-핫 인코딩
    const xs = tf.stack(
      dataFromApi.map((item: any) =>
        oneHotEncode(preprocessText(item.input), maxlengthInput)
      )
    );
    const ys = tf.stack(
      dataFromApi.map((item: any) =>
        oneHotEncode(preprocessText(item.output), maxlengthOutput)
      )
    );

    // 모델 학습
    await model.fit(xs, ys, {
      batchSize,
      epochs,
      // 콜백 추가
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`Epoch: ${epoch + 1} - Loss: ${logs?.loss}`);
          trainLogs.push(logs);
        }
      }
    });

    // setText("hello");
    // predict("hello");
  };

  const predict = (inputText: string) => {
    if (!model) {
      return;
    }

    // 입력 텍스트 전처리 및 원-핫 인코딩
    const input = oneHotEncode(preprocessText(inputText), maxlengthInput);
    const inputData = tf.tensor([input]);

    // 모델 추론
    const prediction = model.predict(inputData) as tf.Tensor;
    const predictionArray = Array.from(prediction.dataSync());

    // 결과 텍스트 생성
    const generatedText = sample(predictionArray, maxlengthOutput);
    setAnswer(generatedText);
  };

  return (
    <div
      style={{
        fontFamily: "sans-serif",
        textAlign: "center",
        paddingTop: "20px"
      }}
    >
      <h1>간단한 자연어 처리 모델</h1>
      <div>
        <button
          onClick={createModel}
          style={{
            background: "#4CAF50",
            border: "none",
            color: "white",
            padding: "15px 32px",
            textAlign: "center",
            textDecoration: "none",
            display: "inline-block",
            fontSize: "16px",
            margin: "4px 10px",
            borderRadius: "12px"
          }}
        >
          모델 생성
        </button>
        <button
          onClick={trainModel}
          style={{
            background: "#2196F3",
            border: "none",
            color: "white",
            padding: "15px 32px",
            textAlign: "center",
            textDecoration: "none",
            display: "inline-block",
            fontSize: "16px",
            margin: "4px 10px",
            borderRadius: "12px"
          }}
        >
          모델 학습
        </button>
      </div>
      <br />
      <div
        style={{
          width: "100%",
          display: "flex",
          justifyContent: "center"
        }}
      >
        <input
          type="text"
          onChange={(e) => setText(e.target.value)}
          value={text}
          style={{
            padding: "12px 20px",
            margin: "8px 0",
            boxSizing: "border-box",
            borderRadius: "4px",
            width: "50%"
          }}
        />
      </div>
      <br />
      <button
        onClick={() => predict(text)}
        style={{
          background: "#008CBA",
          border: "none",
          color: "white",
          padding: "15px 32px",
          textAlign: "center",
          textDecoration: "none",
          display: "inline-block",
          fontSize: "16px",
          margin: "4px 10px",
          borderRadius: "12px"
        }}
      >
        예측하기
      </button>
      <div style={{ fontSize: "20px", marginTop: "20px" }}>
        <strong>답변:</strong> <span>{answer}</span>
      </div>
    </div>
  );
};

export default NaturalLanguageProcessing;

/*
유의미한 결과가 나오지 않는 원인은 여러 가지일 수 있어요. 몇 가지 가능한 이유를 정리해 봤습니다.

데이터셋 크기: 90개의 데이터 포인트는 신경망을 학습시키기에 다소 부족할 수 있습니다. 
특히 복잡한 자연어 처리 문제의 경우, 데이터셋 크기가 수천 또는 수백만 개가 되어야 좀 더 견고한 학습이 가능합니다.

데이터 전처리: 전처리 단계에서 문제가 발생했을 수도 있습니다. 이 경우, 원-핫 인코딩 또는 preprocessText 함수에 문제가 있으니 확인해 보세요.

모델 복잡도: 사용하고 있는 모델의 경우 LSTM 층이 두 개이지만, 문제를 해결하기에 충분한 복잡성을 가지지 못할 수 있습니다. 
층을 더 추가하거나 다양한 파라미터를 변경해 볼 수 있습니다.

과적합(Overfitting): 데이터셋이 작으면서 epoch이 50으로 설정된 경우, 모델이 학습 데이터셋에만 과도하게 적합될 가능성이 있습니다. 
이런 경우, 검증 데이터셋의 결과를 확인하거나 조기 종료(Early Stopping)와 같은 방법으로 파라미터를 수정해 보세요.

학습 속도와 최적화: 학습률(learning rate)이 너무 크거나 작으면 최적의 가중치에 도달하기 어려울 수 있습니다. 
다양한 학습률을 시도해 보세요. 최적화기(optimizer)도 "adam" 외에 다른 것을 시도해 볼 수 있습니다.

해결책으로는 먼저 더 많은 데이터를 확보하여 훈련 데이터셋을 늘리는 것이 좋고, 
데이터 전처리 및 모델 구조에 대해 더 조사해서 적절한 변경을 적용해 보세요. 또한 다양한 학습률과 최적화 알고리즘을 사용해 실험해 보는 것이 좋습니다.
*/
