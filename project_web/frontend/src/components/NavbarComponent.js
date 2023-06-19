import { Link, useLocation, useNavigate } from "react-router-dom";
import { Divv, RowFlex, Nav } from "./StyledComponents";
import {
  Divider,
  ListItemText,
  Menu,
  MenuItem,
  Tooltip,
  Typography,
} from "@mui/material";
import { useState } from "react";

import useStore from "./store";
import Terminal from "react-terminal-ui";

export default function NavbarComponent() {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [menuAnchorElement, setMenuAnchorElement] = useState(null);

  const terminalFontSize = useStore((state) => state.terminalFontSize);

  const increaseTerminalFontSize = useStore(
    (state) => state.increaseTerminalFontSize
  );
  const decreaseTerminalFontSize = useStore(
    (state) => state.decreaseTerminalFontSize
  );

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

          <Divv>
            <Link
              style={{
                textDecoration: "none",
                color: "black",
                float: "right",
              }}
              onClick={(event) => {
                setMenuAnchorElement(event.currentTarget);
                setSettingsOpen(true);
              }}
            >
              settings
            </Link>
            <div style={{ display: "flex", flexDirection: "row" }}>
              <Menu
                anchorEl={menuAnchorElement}
                open={settingsOpen}
                onClose={() => setSettingsOpen(false)}
                anchorOrigin={{
                  vertical: "top",
                  horizontal: "right",
                }}
              >
                <MenuItem>current font size: {terminalFontSize}</MenuItem>
                <Divider />
                <MenuItem onClick={() => increaseTerminalFontSize()}>
                  increase terminal font size
                </MenuItem>
                <MenuItem onClick={() => decreaseTerminalFontSize()}>
                  decrease terminal font size
                </MenuItem>
              </Menu>

              {/* {settingsOpen && (
                <Terminal id="eda-terminal" name="python outputs">
                  <span style={{ fontSize: terminalFontSize }}>
                    {">>> hello there"}
                    {">>> hello there"}
                    {">>> hello there"}
                    {">>> hello there"}
                    {">>> hello there"}
                    {">>> hello there"}
                    <br></br>
                  </span>
                </Terminal>
              )} */}
            </div>
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
