import { Link } from "react-router-dom";
import { Divv, RowFlex, Nav } from "./StyledComponents";
import { useEffect } from "react";

export default function NavbarComponent() {
  useEffect(() => {
    console.log(window.location);

    console.log(window.location.href);
  }, []);

  return (
    <>
      <Nav>
        <RowFlex>
          <Divv size="40px" margin="35px">
            <Link
              to="/"
              style={{
                textDecoration: "none",
                color: window.location.pathname === "/" ? "white" : "black",
              }}
            >
              APESDNM
            </Link>
          </Divv>

          <Divv>
            <Link
              to="/upload"
              style={{
                textDecoration: "none",
                color: window.location.href.includes("upload")
                  ? "white"
                  : "black",
              }}
              onClick={() => {
                console.log(window.location);
              }}
            >
              Upload
            </Link>
          </Divv>

          <Divv>
            <Link
              to="/about"
              style={{
                textDecoration: "none",
                color: window.location.href.includes("about")
                  ? "white"
                  : "black",
              }}
            >
              About
            </Link>
          </Divv>

          <Divv>
            <Link
              to="/contact"
              style={{
                textDecoration: "none",
                color: window.location.href.includes("contact")
                  ? "white"
                  : "black",
              }}
            >
              Contact
            </Link>
          </Divv>
        </RowFlex>
      </Nav>
    </>
  );
}
