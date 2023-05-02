import { Link } from "react-router-dom";
import { Divv } from "./StyledComponents";

export default function Home() {
  return (
    <Divv>
      Hello there, tarnished. It seems you are maidenless. Get maidens{" "}
      <Link to="https://store.steampowered.com/app/1245620/ELDEN_RING/">
        here
      </Link>
      .
    </Divv>
  );
}
