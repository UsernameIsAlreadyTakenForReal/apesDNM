import { Backdrop, Button, CircularProgress, TextField } from "@mui/material";
import { Divv, RowFlex, TextFieldFlex } from "./StyledComponents";
import { useState, useEffect } from "react";

import Terminal, { ColorMode, TerminalOutput } from "react-terminal-ui";
// Make sure to run npm run install-peers after npm install so peer dependencies are also installed.

import io from "socket.io-client";

export default function RoutesComponent() {
  const [terminalText, setTerminalText] = useState(">>> hello world");

  async function getSSEs() {
    const response = await fetch("http://127.0.0.1:5000/testing", {
      method: "get",
    });

    const textResponse = await response.text();
    console.log(textResponse);
  }

  useEffect(() => {
    const socket = io("http://127.0.0.1:5000");
    console.log("socket created");

    socket.on("message", (data) => {
      console.log(data);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  return (
    <>
      <div
        style={{
          margin: "25px",
          width: "auto",
        }}
      >
        <Terminal name="python terminal mwahaha">{terminalText}</Terminal>
      </div>

      <Divv>
        <Button
          style={{ backgroundColor: "black", color: "white" }}
          onClick={() => getSSEs()}
        >
          Click me
        </Button>
      </Divv>
    </>
  );
}
