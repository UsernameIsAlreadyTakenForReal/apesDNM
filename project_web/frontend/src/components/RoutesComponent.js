import { Route, Routes } from "react-router-dom";

import About from "./About";
import Contact from "./Contact";
import UploadComponent from "./UploadComponent";
import Home from "./Home";
import NotFound from "./NotFound";
import TestingComponent from "./TestingComponent";
import Results from "./ResultsComponent";

export default function RoutesComponent() {
  return (
    <>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/upload" element={<UploadComponent />} />
        <Route path="/about" element={<About />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="/results" element={<Results />} />
        <Route path="/testing" element={<TestingComponent />} />

        <Route path="*" element={<NotFound />} />
      </Routes>
    </>
  );
}
