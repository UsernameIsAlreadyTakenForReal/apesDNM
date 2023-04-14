import { UploadButton, UploadDropzone } from "react-uploader";
import { Uploader } from "uploader";
import { Divv, TextFieldDivv, RowFlex } from "./StyledComponents";
import { useState, useEffect } from "react";
import { TextField, Button } from "@material-ui/core";

const BASE_URL = process.env.REACT_APP_BACKEND;

export default function UploadComponent() {
  const [hover1, setHover1] = useState(false);
  const [hover2, setHover2] = useState(false);
  const [hover3, setHover3] = useState(false);

  const [text, setText] = useState("");
  const [apiData, setApiData] = useState({ value: -1 });

  const [state, setState] = useState({
    file: null,
  });

  const FETCH = {
    GET: "get",
    POST: "post",
  };

  async function playFetch(method, route, setFunction) {
    await fetch(BASE_URL + route, {
      method: method,
      headers: { "Content-Type": "application/json" },
    })
      .then((response) => response.json())
      .then((response) => {
        setFunction(response);
      });
  }

  async function playFetchWithData(method, route, data, setFunction) {
    await fetch(BASE_URL + route, {
      method: method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    })
      .then((response) => response.json())
      .then((response) => {
        setFunction(response);
      });
  }

  async function onFileUpload() {
    const formData = new FormData();
    formData.append("file", state.file);

    let body = { file: state.file };

    await fetch(BASE_URL + "upload", {
      method: "post",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then((response) => response.json())
      .then((response) => {
        setText(response);
      });
  }

  function onFileChange(event) {
    setState({ file: event.target.files[0] });
  }

  useEffect(() => {
    setText(apiData.value);
  }, [apiData]);

  return (
    <>
      {false ? (
        <>
          <RowFlex justify="left">
            <Divv>
              <Button
                style={{
                  background: hover1 === false ? "black" : "orange",
                  color: hover1 === false ? "white" : "black",
                  fontWeight: "bold",
                }}
                variant="contained"
                color="primary"
                size="large"
                onClick={() => {
                  playFetch(FETCH.GET, "morbin", setApiData);
                }}
                onMouseEnter={() => {
                  setHover1(true);
                }}
                onMouseLeave={() => {
                  setHover1(false);
                }}
              >
                FETCH GET
              </Button>
            </Divv>

            <Divv>
              <Button
                style={{
                  background: hover2 === false ? "black" : "orange",
                  color: hover2 === false ? "white" : "black",
                  fontWeight: "bold",
                }}
                variant="contained"
                color="primary"
                size="large"
                onClick={() => {
                  setText("");
                  document.getElementById("valueField").value = 0;
                }}
                onMouseEnter={() => {
                  setHover2(true);
                }}
                onMouseLeave={() => {
                  setHover2(false);
                }}
              >
                CLEAR
              </Button>
            </Divv>
          </RowFlex>

          <RowFlex justify="left">
            <Divv>
              <Button
                style={{
                  background: hover3 === false ? "black" : "orange",
                  color: hover3 === false ? "white" : "black",
                  fontWeight: "bold",
                }}
                variant="contained"
                color="primary"
                size="large"
                onClick={() => {
                  const body = {
                    value: document.getElementById("valueField").value,
                  };

                  playFetchWithData(FETCH.POST, "double", body, setApiData);
                }}
                onMouseEnter={() => {
                  setHover3(true);
                }}
                onMouseLeave={() => {
                  setHover3(false);
                }}
              >
                FETCH POST with this data:
              </Button>
            </Divv>

            <Divv>
              <TextFieldDivv>
                <TextField variant="outlined" id="valueField" />
              </TextFieldDivv>
            </Divv>
          </RowFlex>

          <RowFlex justify="left">
            <Divv>from backend: {text}</Divv>

            <Divv>apiData.value: {apiData.value}</Divv>
          </RowFlex>
        </>
      ) : (
        <></>
      )}
      <Divv margin="30px">
        <input
          type="file"
          onChange={(event) => {
            onFileChange(event);
          }}
        />
        <button
          onClick={() => {
            onFileUpload();
          }}
        >
          Upload
        </button>
      </Divv>
    </>
  );
}

// -------------------------- EXTRAS --------------------------

// const options = {
//   multi: true,
//   styles: {
//     margin: "120px",
//   },
// };

// const uploader = Uploader({
//   apiKey: "free",
// });

// const uploaderOptions = {
//   multi: true,
//   showFinishButton: true,
//   styles: {
//     colors: {
//       primary: "#377dff",
//     },
//   },
// };

// const MyDropzone = ({ setFiles }) => (
//   <UploadDropzone
//     uploader={uploader}
//     options={uploaderOptions}
//     onUpdate={(files) =>
//       console.log(`Files: ${files.map((x) => x.fileUrl).join("\n")}`)
//     }
//     onComplete={setFiles}
//     width="600px"
//     height="375px"
//   >
//     Upload a dataset
//   </UploadDropzone>
// );
