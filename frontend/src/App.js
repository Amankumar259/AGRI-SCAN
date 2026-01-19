import React, { useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { motion } from "framer-motion";
import "@/App.css";
import Home from "./pages/Home";
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
