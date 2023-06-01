import { Backdrop, Button, CircularProgress, TextField } from "@mui/material";
import { Divv, RowFlex, TextFieldFlex } from "./StyledComponents";
import { useState, useEffect } from "react";

import Terminal, { ColorMode, TerminalOutput } from "react-terminal-ui";
// Make sure to run npm run install-peers after npm install so peer dependencies are also installed.

export default function RoutesComponent() {
  const [terminalText, setTerminalText] = useState("hello world");

  async function getSSEs() {
    console.log("getSSEs() called");
    const response = await fetch("http://127.0.0.1:5000/" + "testingSSEs", {
      method: "get",
    });

    const textResponse = await response.text();
    console.log(textResponse);
    console.log("getSSEs() ended");
  }

  useEffect(() => {
    const source = new EventSource("http://127.0.0.1:5000/stream");
    console.log("sse started");

    source.addEventListener("message", (event) => {
      const message = event.data;
      console.log("event happened");
      // Handle SSE message, e.g., update component state
      console.log("Received SSE message:", message);
    });

    return () => {
      source.close();
      console.log("sse closed");
    };
  }, []);

  return (
    <>
      {/* <div
        style={{
          margin: "25px",
          width: "auto",
        }}
      >
        <Terminal name="python terminal mwahaha">{terminalText}</Terminal>
      </div> */}

      <Divv>
        <Button
          style={{ backgroundColor: "black", color: "white" }}
          onClick={() => {
            getSSEs();
          }}
        >
          Click me
        </Button>
      </Divv>
    </>
  );
}
