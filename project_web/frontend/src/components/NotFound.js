import { Link } from "react-router-dom";
import { Divv } from "./StyledComponents";

export default function NotFound() {
  return (
    <Divv>
      it seems that you have reached a wrong page. go back{" "}
      <Link to="/">here</Link>.
    </Divv>
  );
}
