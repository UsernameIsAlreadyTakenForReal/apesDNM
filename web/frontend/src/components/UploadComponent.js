import { Divv, TextFieldDivv, RowFlex } from "./StyledComponents";
import { useState, useEffect } from "react";
import { TextField, Button } from "@material-ui/core";

const BASE_URL = process.env.REACT_APP_BACKEND;

export default function UploadComponent() {
  const [hover1, setHover1] = useState(false);
  const [text1, setText1] = useState("");

  const [hover2, setHover2] = useState(false);
  const [text2, setText2] = useState("");

  const [hover3, setHover3] = useState(false);
  const [text3, setText3] = useState("");

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
    const jsonData = await response.json();
    setText1(jsonData.message);
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

    const jsonData = await response.json();
    setText2(jsonData.message);
    console.log(jsonData);
  }

  async function onClick3Handle() {
    const response = await fetch(BASE_URL + "morbin", {
      method: "get",
    });
    const jsonData = await response.json();
    setText1(jsonData.message);
  }

  return (
    <>
      <RowFlex justify="left">
        <Divv>
          <Button
            style={{
              background: "black",
              color: "white",
              fontWeight: "bold",
            }}
            variant="contained"
            color="primary"
            size="large"
            onClick={() => {
              clearAllData();
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
              background: hover2 === false ? "black" : "orange",
              color: hover2 === false ? "white" : "black",
              fontWeight: "bold",
            }}
            variant="contained"
            color="primary"
            size="large"
            onClick={() => {
              onClick3Handle();
            }}
            onMouseEnter={() => {
              setHover2(true);
            }}
            onMouseLeave={() => {
              setHover2(false);
            }}
          >
            FETCH UPL
          </Button>
        </Divv>

        <input type="file"></input>

        <Divv>from backend: {text3}</Divv>
      </RowFlex>
    </>
  );
}
