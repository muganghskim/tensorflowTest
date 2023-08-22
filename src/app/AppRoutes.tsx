import * as React from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Home from "../pages/Home";
import Diabetes from "../pages/Diabetes";
import BTCPricePredictor from "../pages/BTCPricePredictor";
import NaturalLanguageProcessing from "../pages/NaturalLanguageProcessing";

function AppRoutes() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/diabetes-prediction" element={<Diabetes />} />
        <Route path="/btc-prediction" element={<BTCPricePredictor />} />
        <Route
          path="/NaturalLanguageProcessing"
          element={<NaturalLanguageProcessing />}
        />
        {/*  추가적인 라우트를 이곳에 작성해주세요  */}
      </Routes>
    </BrowserRouter>
  );
}

export default AppRoutes;
