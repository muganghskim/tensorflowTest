import React from "react";
import AppRoutes from "./app/AppRoutes";
import { RecoilRoot } from "recoil";

function App() {
  return (
    <div className="App">
      <RecoilRoot>
        <AppRoutes />
      </RecoilRoot>
    </div>
  );
}

export default App;
