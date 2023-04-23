import { Divv, TextFieldDivv, RowFlex } from "./StyledComponents";
import { useState, useEffect } from "react";
import { TextField, Button } from "@material-ui/core";

const BASE_URL = process.env.REACT_APP_BACKEND;

export default function UploadComponent() {
  const [hover0, setHover0] = useState(false);

  const [hover1, setHover1] = useState(false);
  const [text1, setText1] = useState("");

  const [hover2, setHover2] = useState(false);
  const [text2, setText2] = useState("");

  const [hover3, setHover3] = useState(false);
  const [text3, setText3] = useState("");

  const [file, setFile] = useState(null);

  function clearAllData() {
    setText1("");
    setText2("");
    setText3("");
    document.getElementById("valueField1").value = "";
    document.getElementById("valueField2").value = "";
  }

  async function onClick1Handle() {
    const response = await fetch(BASE_URL + "morbin", {
      method: "get",
    });
    const jsonData = await response.text();
    setText1(jsonData);
    console.log(jsonData);
  }

  async function onClick2Handle() {
    let data = {
      value1: document.getElementById("valueField1").value
        ? document.getElementById("valueField1").value
        : "0",
      value2: document.getElementById("valueField2").value
        ? document.getElementById("valueField2").value
        : "0",
    };

    console.log(data);

    data = JSON.stringify(data);

    const response = await fetch(BASE_URL + "double", {
      method: "post",
      headers: { "Content-Type": "application/json" },
      body: data,
    });

    const jsonData = await response.text();
    setText2(jsonData);
    console.log(jsonData);
  }

  async function onUploadHandler() {
    if (!file) return;
    setText3("loading...");
    console.log("file is " + file.name);
    console.log("type is " + file.type);

    let reader = new FileReader();
    reader.readAsText(file);

    console.log(reader);

    let fd = new FormData();
    fd.append("file", reader);

    const response = await fetch(BASE_URL + "upload", {
      method: "post",
      body: fd,
    });

    const data = await response.text();
    setText3("OK");
    console.log(data);
  }

  return (
    <>
      <RowFlex justify="left">
        <Divv>
          <Button
            style={{
              background: hover0 === false ? "black" : "orange",
              color: hover0 === false ? "white" : "black",
              fontWeight: "bold",
            }}
            variant="contained"
            color="primary"
            size="large"
            onClick={() => {
              clearAllData();
            }}
            onMouseEnter={() => {
              setHover0(true);
            }}
            onMouseLeave={() => {
              setHover0(false);
            }}
          >
            CLEAR ALL
          </Button>
        </Divv>
        <Divv>This is the upload page.</Divv>
      </RowFlex>

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
              onClick1Handle();
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

        <Divv>from backend: {text1}</Divv>
      </RowFlex>

      <RowFlex justify="left">
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
              onClick2Handle();
            }}
            onMouseEnter={() => {
              setHover2(true);
            }}
            onMouseLeave={() => {
              setHover2(false);
            }}
          >
            FETCH SUM
          </Button>
        </Divv>

        <Divv margin="0px">
          <TextFieldDivv>
            <TextField variant="outlined" id="valueField1" />
          </TextFieldDivv>

          <TextFieldDivv>
            <TextField variant="outlined" id="valueField2" />
          </TextFieldDivv>
        </Divv>

        <Divv>from backend: {text2}</Divv>
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
              console.clear();
              onUploadHandler();
            }}
            onMouseEnter={() => {
              setHover3(true);
            }}
            onMouseLeave={() => {
              setHover3(false);
            }}
          >
            FETCH UPL
          </Button>
        </Divv>

        <input
          type="file"
          onChange={(event) => setFile(event.target.files[0])}
        ></input>

        <Divv>from backend: {text3}</Divv>
      </RowFlex>
    </>
  );
}
