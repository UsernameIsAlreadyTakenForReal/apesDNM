import { Divv, TextFieldDivv, RowFlex } from "./StyledComponents";
import { useState, useEffect } from "react";
import { TextField, Button } from "@material-ui/core";

const BASE_URL = process.env.REACT_APP_BACKEND;

export default function UploadComponent() {
  const [hover1, setHover1] = useState(false);
  const [text1, setText1] = useState("");

  function clearAllData() {
    setText1("");
  }

  function onClick1Handle() {
    fetch(BASE_URL + "morbin", {
      method: "get",
    }).then((data) => {
      console.log(data);
      setText1(data);
    });
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
    </>
  );
}
