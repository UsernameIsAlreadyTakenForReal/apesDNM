import { Divv } from "./StyledComponents";
import { Link } from "react-router-dom";

export default function About() {
  return (
    <Divv>
      this is the 'about' page of our{" "}
      <Link
        style={{ color: "black", textDecoration: "none" }}
        target="_blank"
        to="https://github.com/UsernameIsAlreadyTakenForReal/apesDNM"
      >
        urban-octo-dizeratie
      </Link>
      . it is a creation of cârlianu radu and pupitru daniel.
      <br /> <br />
      <Divv margin="0px" size="22.5px">
        our goal is to create a self-maintaing and sustaining application which
        is capable of receiving an input (a dataset) and choosing which
        intelligent method is best suited to detect anomalies on that given
        input.
      </Divv>
      <br></br>
      {/* <Divv margin="0px" size="15px">
        *apesdnm = acest proiect este supervizat de norm macdonald
      </Divv> */}
      <Divv margin="0px" size="22.5px">
        the all lowercase was inspired by ee cummings' often stylised spelling
        of his own name
      </Divv>
    </Divv>
  );
}
