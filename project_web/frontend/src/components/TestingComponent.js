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

import { useState } from "react";
import { Divv } from "./StyledComponents";

import lstmSVG from "../lstm.svg";
import cnnSVG from "../cnn.svg";

import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
} from "@mui/material";

const BASE_URL = process.env.REACT_APP_BACKEND;

function TestingComponent() {
  const [dialogText, setDialogText] = useState(
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et egestas elit. Pellentesque eleifend justo vel lectus mattis ullamcorper. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Donec eu tincidunt lorem. Quisque laoreet quam at nisl aliquet, et ultrices lacus placerat. Vestibulum dui nunc, ultricies rhoncus maximus non, lobortis sit amet enim. Integer at ultrices enim. Suspendisse mauris justo, suscipit ut pulvinar quis, malesuada et magna."
  );

  const [showEDA, setShowEDA] = useState(true);
  const [edaButtonHover, setEDAButtonHover] = useState(false);

  return (
    <>
      {showEDA && (
        <Dialog open={true} maxWidth="xl" fullWidt={true}>
          <DialogTitle style={{ fontWeight: "bold" }}>
            {"are you happy with your dataset motherfucker?"}
          </DialogTitle>
          <DialogContent>
            <DialogContentText>
              we ran some test and found this out: your dataset fucking sucks
            </DialogContentText>
            <img src={lstmSVG} />
            <DialogContentText>
              see this thing above? it's got some smart things in those neurons
              unlike you you monumental degenerate
            </DialogContentText>
          </DialogContent>
          <DialogActions>
            <Button
              style={{
                background: edaButtonHover === false ? "black" : "orange",
                color: edaButtonHover === false ? "white" : "black",
                fontWeight: "bold",
              }}
              variant="contained"
              color="primary"
              size="large"
              onMouseEnter={() => setEDAButtonHover(true)}
              onMouseLeave={() => setEDAButtonHover(false)}
            >
              got it, sorry
            </Button>
          </DialogActions>
        </Dialog>
      )}
    </>
  );
}

export default TestingComponent;
