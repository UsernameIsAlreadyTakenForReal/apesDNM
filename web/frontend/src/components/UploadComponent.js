import { UploadButton, UploadDropzone } from "react-uploader";
import { Uploader } from "uploader";
import { Divv, RowFlex } from "./StyledComponents";
import { useState } from "react";
import { Button } from "@material-ui/core";

const BASE_URL = process.env.REACT_APP_BACKEND;

export default function UploadComponent() {
  const [text, setText] = useState("");
  const [hover1, setHover1] = useState(false);
  const [hover2, setHover2] = useState(false);

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

  async function playFetch(method, route, setFunction) {
    await fetch(BASE_URL + route, {
      method: method,
      headers: { "Content-Type": "application/json" },
    })
      .then((response) => response.json())
      .then((response) => {
        setFunction(response);
        return response;
      });
  }

  function playFlask() {
    fetch(BASE_URL + "testing").then((res) =>
      res.json().then((data) => {
        setText(data);
      })
    );
  }

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
              // playFetch(FETCH.GET, "testing", setText);
              // playFlask();
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
              setText("HOLD UP I AM STILL TESTING");
            }}
            onMouseEnter={() => {
              setHover2(true);
            }}
            onMouseLeave={() => {
              setHover2(false);
            }}
          >
            FETCH POST
          </Button>
        </Divv>
      </RowFlex>

      <Divv>From backend: {text === "" ? "" : '"' + text + '"'}</Divv>
      <Divv>
        <MyDropzone />
      </Divv>
    </>
  );
}
