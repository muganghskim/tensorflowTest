import * as React from "react";
import "../../src/assets/scss/home.css";

const Home: React.FC = () => {
  return (
    <div className="Home">
      <h1>머신러닝 체험</h1>
      <ul className="ml-links">
        <li>
          <a href="/diabetes-prediction" className="ml-link">
            <div>
              <h2>당뇨병 예측</h2>
              <p>건강지표를 확인하여 당뇨병을 인식하고 예측하십시오.</p>
            </div>
          </a>
        </li>
        <li>
          <a href="/NaturalLanguageProcessing" className="ml-link">
            <div>
              <h2>자연어 처리</h2>
              <p>텍스트의 감정을 분석하십시오.</p>
            </div>
          </a>
        </li>
        <li>
          <a href="/object-detection" className="ml-link">
            <div>
              <h2>객체 탐지</h2>
              <p>이미지에서 다양한 객체를 탐지하십시오.</p>
            </div>
          </a>
        </li>
        {/* 추가적인 머신러닝 체험 기능들을 여기에 추가하세요. */}
      </ul>
      <ul className="ml-links">
        <li>
          <a href="/btc-prediction" className="ml-link">
            <div>
              <h2>BTC 예측</h2>
              <p>비트코인가격을 분석하여 향후 가격예측</p>
            </div>
          </a>
        </li>
        {/* 추가적인 머신러닝 체험 기능들을 여기에 추가하세요. */}
      </ul>
    </div>
  );
};

export default Home;
