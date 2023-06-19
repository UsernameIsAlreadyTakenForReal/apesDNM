import { Divv } from "./StyledComponents";

import apesdnmLogo01 from "../apesdnm01.png";
import apesdnmLogo02 from "../apesdnm02.png";

import apesLogo from "../apes-logo.png";

import { useState } from "react";

export default function Home() {
  const [imgSrc, setImgSrc] = useState(apesdnmLogo01);
  return (
    <Divv>
      <Divv
        top="30px"
        size="40px"
        style={{ display: "flex", justifyContent: "center" }}
      >
        welcome to our lowercase neural-network (somewhat) smart system!{" "}
      </Divv>
      <div
        style={{ display: "flex", justifyContent: "center", marginTop: "35px" }}
      >
        <img
          src={imgSrc}
          width="800"
          height="450"
          // onMouseEnter={() => setImgSrc(apesdnmLogo02)}
          // onMouseLeave={() => setImgSrc(apesdnmLogo01)}
        />
      </div>

      {/* <Divv top="40px" style={{ display: "flex", justifyContent: "center" }}>
        navigate over to upload at the top
      </Divv> */}
    </Divv>
  );
}
