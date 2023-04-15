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

  const [file, setFile] = useState();

  const FETCH = {
    GET: "get",
    POST: "post",
  };

  async function playFetch(method, route, setFunction) {
    await fetch(BASE_URL + route, {
      method: method,
      // headers: { "Content-Type": "application/json" },
    })
      // .then((response) => response.json())
      .then((response) => {
        console.log(response);
        // setFunction(response);
      });
  }

  async function playFetchWithData(method, route, data, setFunction) {
    await fetch(BASE_URL + route, {
      method: method,
      body: JSON.stringify(data),
      // headers: { "Content-Type": "application/json" },
    })
      .then((response) => response.json())
      .then((response) => {
        setFunction(response);
      });
  }

  async function onFileUpload() {
    if (!file) return;

    let fd = new FormData();
    fd.append("file", file);

    await fetch(BASE_URL + "upload", {
      method: "post",
      headers: { "content-Type": file.type, "content-length": `${file.size}` },
      body: file,
    })
      // .then((response) => response.json())
      .then((data) => {
        console.log("data from api is");
        console.log(data);
      })
      .catch((err) => {
        console.log(err);
      });
  }

  function onFileChange(event) {
    let file = event.target.files[0];
    setFile(file);
  }

  useEffect(() => {
    setText(apiData.msg);
  }, [apiData]);

  return (
    <>
      {true ? (
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
                  playFetch(FETCH.GET, "morbin", setText);
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
                  console.clear();
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
        </>
      ) : (
        <></>
      )}

      <RowFlex justify="left">
        <Divv>from backend: {text}</Divv>

        {/* <Divv>apiData.value: {apiData.value}</Divv> */}
      </RowFlex>

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

// -------------------------- EXTRAS #1 --------------------------

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

// -------------------------- EXTRAS #2 --------------------------

// async function onFileUpload() {
//   const formData = new FormData();
//   formData.append("file", file.file);

//   await fetch(BASE_URL + "upload", {
//     method: "post",
//     // headers: { "Content-Type": "application/json" },
//     body: file.file,
//   })
//     .then((response) => response.json())
//     .then((response) => {
//       setApiData(response);
//     });
// }
