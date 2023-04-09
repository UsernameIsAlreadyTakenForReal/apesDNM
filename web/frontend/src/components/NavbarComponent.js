import styled from "styled-components";
import { Link } from "react-router-dom";
import { useState } from "react";

import { Divv } from "./StyledComponents";

export default function NavbarComponent() {
  const [route, setRoute] = useState("");

  const Nav = styled.nav`
    background: grey;
    height: 80px;
    display: flex;
    justify-content: space-between;
  `;

  const RowFlex = styled.div`
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
  `;

  return (
    <>
      <Nav>
        <RowFlex>
          <Divv size="40px" margin="35px">
            <Link
              to="/"
              style={{
                textDecoration: "none",
                color: route === "" ? "white" : "black",
              }}
              onClick={() => {
                setRoute("");
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
                color: route.includes("upload") ? "white" : "black",
              }}
              onClick={() => {
                setRoute("upload");
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
                color: route.includes("about") ? "white" : "black",
              }}
              onClick={() => {
                setRoute("about");
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
                color: route.includes("contact") ? "white" : "black",
              }}
              onClick={() => {
                setRoute("contact");
              }}
            >
              Contact
            </Link>
          </Divv>

          <Divv>
            <Link
              style={{
                textDecoration: "none",
                color: "black",
              }}
              onClick={() => {
                console.clear();
              }}
            >
              Clear console
            </Link>
          </Divv>
        </RowFlex>
      </Nav>
    </>
  );
}
