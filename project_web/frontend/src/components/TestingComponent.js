// import { Backdrop, Button, CircularProgress, TextField } from "@mui/material";
// import { Divv, RowFlex, TextFieldFlex } from "./StyledComponents";
// import { useState, useEffect } from "react";

// import Terminal, { ColorMode, TerminalOutput } from "react-terminal-ui";
// // Make sure to run npm run install-peers after npm install so peer dependencies are also installed.

// export default function RoutesComponent() {
//   const [terminalText, setTerminalText] = useState(">>> hello world");
//   const [alreadyConnected, setAlreadyConnected] = useState(false);

//   async function testing() {
//     const response = await fetch("http://127.0.0.1:5000/testing", {
//       method: "get",
//     });

//     const textResponse = await response.text();
//     console.log(textResponse);
//   }

//   return (
//     <>
//       <div
//         style={{
//           margin: "25px",
//           width: "auto",
//         }}
//       >
//         <Terminal name="python terminal mwahaha">{terminalText}</Terminal>
//       </div>

//       <Divv>
//         <Button
//           style={{
//             backgroundColor: "black",
//             color: "white",
//           }}
//           onClick={() => {
//             console.log(window.location.href);
//           }}
//         >
//           Click me
//         </Button>
//       </Divv>
//     </>
//   );
// }

// ###########################################################################

import { io } from "socket.io-client";
import { useEffect, useState } from "react";
import { Divv } from "./StyledComponents";
import { Button } from "@mui/material";
import WebSocketComponent from "./WebSocketComponent";

const BASE_URL = process.env.REACT_APP_BACKEND;

function TestingComponent() {
  const [output, setOutput] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleOutputUpdated = (updatedOutput) => {
    setOutput(updatedOutput);
  };

  async function handleRequest() {
    setLoading(true);

    const response = await fetch(BASE_URL + "testing", {
      method: "get",
    });

    const data = await response.text();
    setLoading(false);
    console.log(data);
  }

  return (
    <>
      <Divv>
        <Button variant="outlined" onClick={() => handleRequest()}>
          Request
        </Button>
        <Divv>from socketio: {output}</Divv>
        <Divv
          style={{
            display: loading ? "" : "none",
          }}
        >
          still waiting for the execution to finish
        </Divv>
      </Divv>

      <WebSocketComponent onOutputUpdated={handleOutputUpdated} />
    </>
  );
}

export default TestingComponent;
