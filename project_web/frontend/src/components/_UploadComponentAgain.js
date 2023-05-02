import { Divv, TextFieldDivv, RowFlex } from "./StyledComponents";
import { useState, useEffect } from "react";
import { TextField, Button } from "@mui/material";

const BASE_URL = process.env.REACT_APP_BACKEND;

export default function UploadComponent() {
  const [hover0, setHover0] = useState(false);

  const [hover1, setHover1] = useState(false);
  const [text1, setText1] = useState("");

  const [hover2, setHover2] = useState(false);
  const [text2, setText2] = useState("");

  const [hover3, setHover3] = useState(false);
  const [text3, setText3] = useState("");
  const [btnText, setBtnText] = useState("FETCH UPL");
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);

  const [file, setFile] = useState(null);

  let count = 0;

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
    if (!file) {
      setText3("Please choose a file");
      return;
    }

    setLoading(true);

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
    setLoading(false);
    setLoaded(true);
    setBtnText("FILE LOADED");
    setText3(data);
    console.log(data);
  }

  useEffect(() => {
    const interval = setInterval(() => {
      if (!loading) {
        return;
      }
      if (count % 12 === 0) setBtnText("╞▰══════╡");
      if (count % 12 === 1) setBtnText("╞═▰═════╡");
      if (count % 12 === 2) setBtnText("╞══▰════╡");
      if (count % 12 === 3) setBtnText("╞═══▰═══╡");
      if (count % 12 === 4) setBtnText("╞════▰══╡");
      if (count % 12 === 5) setBtnText("╞═════▰═╡");
      if (count % 12 === 6) setBtnText("╞══════▰╡");
      if (count % 12 === 7) setBtnText("╞═════▰═╡");
      if (count % 12 === 8) setBtnText("╞════▰══╡");
      if (count % 12 === 9) setBtnText("╞═══▰═══╡");
      if (count % 12 === 10) setBtnText("╞══▰════╡");
      if (count % 12 === 11) setBtnText("╞═▰═════╡");
      count = count + 1;
    }, 75);

    return () => clearInterval(interval);
  }, [loading]);

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
              background: loading
                ? "orange"
                : loaded
                ? "green"
                : !hover3
                ? "black"
                : "orange",
              color: loading ? "black" : hover3 === false ? "white" : "black",
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
            {btnText}
          </Button>
        </Divv>

        <input
          type="file"
          onChange={(event) => {
            setFile(event.target.files[0]);
            setBtnText("FETCH UPL");
            setLoading(false);
            setLoaded(false);
            setText3("");
          }}
        ></input>

        <Divv>from backend: {text3}</Divv>
      </RowFlex>

      <RowFlex justify="left">
        <Divv>
          <Button
            style={{
              background: loading
                ? "orange"
                : loaded
                ? "green"
                : !hover3
                ? "black"
                : "orange",
              color: loading ? "black" : hover3 === false ? "white" : "black",
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
            UPL 2
          </Button>
        </Divv>

        <input
          type="file"
          onChange={(event) => {
            setFile(event.target.files[0]);
            setBtnText("FETCH UPL");
            setLoading(false);
            setLoaded(false);
            setText3("");
          }}
        ></input>

        <Divv>from backend: {text3}</Divv>
      </RowFlex>
    </>
  );
}
