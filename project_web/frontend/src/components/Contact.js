import { Link } from "react-router-dom";
import { Divv } from "./StyledComponents";

export default function Contact() {
  return (
    <Divv size="22.5px">
      cîșlianu (cârlianu) radu
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
      dumitru (pupitru) daniel
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
