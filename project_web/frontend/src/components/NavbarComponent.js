import { Link, useLocation, useNavigate } from "react-router-dom";
import { Divv, RowFlex, Nav } from "./StyledComponents";
import { useEffect, useState } from "react";

export default function NavbarComponent() {
  const navigate = useNavigate();
  return (
    <>
      <Nav>
        <RowFlex>
          <Divv size="42.5px" margin="50px" style={{ padding: "10px" }}>
            <Link
              to="/"
              style={{
                textDecoration: "none",
                color: useLocation().pathname === "/" ? "white" : "black",
              }}
            >
              apesdnm
            </Link>
          </Divv>

          <Divv>
            <Link
              to="/upload"
              style={{
                textDecoration: "none",
                color: useLocation().pathname === "/upload" ? "white" : "black",
              }}
              onClick={() => {
                navigate("/upload");
                navigate(0);
              }}
            >
              upload
            </Link>
          </Divv>

          <Divv>
            <Link
              to="/about"
              style={{
                textDecoration: "none",
                color: useLocation().pathname === "/about" ? "white" : "black",
              }}
            >
              about
            </Link>
          </Divv>

          <Divv>
            <Link
              to="/contact"
              style={{
                textDecoration: "none",
                color:
                  useLocation().pathname === "/contact" ? "white" : "black",
              }}
            >
              contact
            </Link>
          </Divv>

          {/* <Divv>
            <Link
              to="/testing"
              style={{
                textDecoration: "none",
                color:
                  useLocation().pathname === "/testing" ? "white" : "black",
              }}
            >
              Testing
            </Link>
          </Divv> */}
        </RowFlex>
      </Nav>
    </>
  );
}
