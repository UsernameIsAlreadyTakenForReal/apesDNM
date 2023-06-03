import { Link } from "react-router-dom";
import { Divv } from "./StyledComponents";

export default function Home() {
  return (
    <Divv>
      hello there, tarnished. it seems you are maidenless. get maidens{" "}
      <Link to="https://store.steampowered.com/app/1245620/ELDEN_RING/">
        here
      </Link>
      .
    </Divv>
  );
}
