import * as React from "react";
import "../../src/assets/scss/home.css";

const Home: React.FC = () => {
  return (
    <div className="Home">
      <h1>머신러닝 체험</h1>
      <ul className="ml-links">
        <li>
          <a href="/diabetes-classification" className="ml-link">
            <div>
              <h2>당뇨병 예측</h2>
              <p>건강지표를 확인하여 당뇨병을 인식하고 예측하십시오.</p>
            </div>
          </a>
        </li>
        <li>
          <a href="/sentiment-analysis" className="ml-link">
            <div>
              <h2>감성 분석</h2>
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
    </div>
  );
};

export default Home;
