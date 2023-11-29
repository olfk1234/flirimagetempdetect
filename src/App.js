import logo from './logo.svg';
import './App.css';
import {Form} from './Form.js';
import React from 'react';
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Explanation } from './Explanation.js';
import { HowToUse } from './HowToUse.js';

// export default function App() {
//   return (
//     <BrowserRouter>
//       <Routes>
//         <Route path="/" element={<Form />}>
//         </Route>
//       </Routes>
//     </BrowserRouter>
//   );
// }
function App() {
  return (
    <Routes>
      <Route path="/" element={<Form />}/>
      <Route path="/explanation" element={<Explanation />}/>
      <Route path="/howtouse" element={<HowToUse />}/>
    </Routes>
  );
}

export default App;
