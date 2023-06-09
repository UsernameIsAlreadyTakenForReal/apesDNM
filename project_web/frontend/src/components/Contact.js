import { Divv } from "./StyledComponents";
import { Link } from "react-router-dom";
import { Grid } from "@mui/material";

export default function Contact() {
  return (
    <Divv size="22.5px">
      cîșlianu{" "}
      <span style={{ textDecoration: "line-through" }}>(cârlianu)</span> radu
      <br />
      sim | 94286 on upb teams | radu.cislianu@gmail.com | (+40) 736-455-655 |{" "}
      <Link
        style={{ color: "black", textDecoration: "none" }}
        target="_blank"
        to="https://github.com/UsernameIsAlreadyTakenForReal"
      >
        github
      </Link>
      <br /> <br />
      dumitru <span style={{ textDecoration: "line-through" }}>
        (pupitru)
      </span>{" "}
      daniel
      <br />
      ssa | 94817 on upb teams | dmtrudaniel@gmail.com | (+40) 725-575-225 |{" "}
      <Link
        style={{ color: "black", textDecoration: "none" }}
        target="_blank"
        to="https://github.com/IsTheUsernameReallyTaken"
      >
        github
      </Link>
    </Divv>
  );
}
