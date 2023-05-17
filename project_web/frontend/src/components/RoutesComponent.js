import { Link, Route, Routes, Redirect, Navigate } from "react-router-dom";
import styled from "styled-components";

import About from "./About";
import Contact from "./Contact";
import UploadComponent from "./UploadComponent";
import Home from "./Home";
import NotFound from "./NotFound";
import _Uploaddd from "./_Uploaddd";
import TestingComponent from "./TestingComponent";

export default function RoutesComponent() {
  return (
    <>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/upload" element={<UploadComponent />} />
        <Route path="/uploaddd" element={<_Uploaddd />} />
        <Route path="/about" element={<About />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="/testing" element={<TestingComponent />} />

        <Route path="*" element={<NotFound />} />
      </Routes>
    </>
  );
}
