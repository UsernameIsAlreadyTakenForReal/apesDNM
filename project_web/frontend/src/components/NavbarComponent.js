import { Link, useLocation } from "react-router-dom";
import { Divv, RowFlex, Nav } from "./StyledComponents";
import { useEffect, useState } from "react";

export default function NavbarComponent() {
  return (
    <>
      <Nav>
        <RowFlex>
          <Divv size="40px" margin="35px">
            <Link
              to="/"
              style={{
                textDecoration: "none",
                color: useLocation().pathname === "/" ? "white" : "black",
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
                color: useLocation().pathname === "/upload" ? "white" : "black",
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
                color: useLocation().pathname === "/about" ? "white" : "black",
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
                color:
                  useLocation().pathname === "/contact" ? "white" : "black",
              }}
            >
              Contact
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
