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

  const FETCH = {
    GET: "get",
    POST: "post",
  };

  const uploader = Uploader({
    apiKey: "free",
  });

  const options = {
    multi: true,
    styles: {
      margin: "120px",
    },
  };

  const uploaderOptions = {
    multi: true,
    showFinishButton: true,
    styles: {
      colors: {
        primary: "#377dff",
      },
    },
  };

  const MyDropzone = ({ setFiles }) => (
    <UploadDropzone
      uploader={uploader}
      options={uploaderOptions}
      onUpdate={(files) =>
        console.log(`Files: ${files.map((x) => x.fileUrl).join("\n")}`)
      }
      onComplete={setFiles}
      width="600px"
      height="375px"
    >
      Upload a dataset
    </UploadDropzone>
  );

  function playFetch(method, route, setFunction) {
    fetch(BASE_URL + route, {
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
      body: data,
    })
      .then((response) => response.json())
      .then((response) => {
        console.log("this is from the fetch function - response");
        console.log(response);

        setFunction(response);
        // setText(response.value);
        // https://stackoverflow.com/questions/54069253/the-usestate-set-method-is-not-reflecting-a-change-immediately

        console.log("this is from the fetch function - apiData");
        console.log(apiData);
      });
  }

  // async function playFetchAsync(method, route, setFunction) {
  //   await fetch(BASE_URL + route, {
  //     method: method,
  //     headers: { "Content-Type": "application/json" },
  //   })
  //     .then((response) => response.json())
  //     .then((response) => {
  //       setFunction(response);
  //     });
  // }

  // function playFlask(setFunction) {
  //   let data = fetch(BASE_URL + "morbin").then((res) =>
  //     res.json().then((data) => {
  //       setFunction(data);
  //     })
  //   );
  //   return data;
  // }

  return (
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

              console.log(body);

              playFetchWithData(
                FETCH.POST,
                "testing",
                JSON.stringify(body),
                setApiData
              );

              console.log(
                "this is fron the onClick function handler - api Data"
              );
              console.log(apiData);

              setText(apiData.value);
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

      <Divv>From backend: {text === "" ? "nothing" : '"' + text + '"'}</Divv>
      <Divv>
        <MyDropzone />
      </Divv>
    </>
  );
}
