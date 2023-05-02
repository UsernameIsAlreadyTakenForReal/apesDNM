import { Link } from "react-router-dom";
import { Divv } from "./StyledComponents";

export default function NotFound() {
  return (
    <Divv>
      It seems that you have reached a wrong page. Go back{" "}
      <Link to="/">here</Link>.
    </Divv>
  );
}
