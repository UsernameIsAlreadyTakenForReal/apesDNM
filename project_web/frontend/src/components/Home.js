import { Divv } from "./StyledComponents";

import apesdnmLogo from "../apesdnm.png";

export default function Home() {
  return (
    <Divv>
      <Divv top="30px" style={{ display: "flex", justifyContent: "center" }}>
        welcome to our lowercase neural-network (somewhat) smart system!{" "}
      </Divv>
      <div
        style={{ display: "flex", justifyContent: "center", marginTop: "35px" }}
      >
        <img src={apesdnmLogo} width="800" height="450" />
      </div>

      {/* <Divv top="40px" style={{ display: "flex", justifyContent: "center" }}>
        navigate over to upload in order at the top
      </Divv> */}
    </Divv>
  );
}
