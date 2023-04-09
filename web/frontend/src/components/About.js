import { Divv } from "./StyledComponents";

export default function About() {
  return (
    <Divv>
      This is the 'about' page of our urban-octo-dizeratie. This is a creation
      by Cârlianu Radu and Pupitru Daniel.
      <br /> <br />
      <Divv margin="0px" size="22.5px">
        Our goal is to create a self-maintaing and sustaining application which
        is capable of receiving an input (a dataset) and chooses which
        intelligent method is best suited to detect anomalies on the given
        input.
      </Divv>
    </Divv>
  );
}
