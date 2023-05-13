import { useEffect, useState } from "react";
import { Divv } from "./StyledComponents";
import {
  Button,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
} from "@mui/material";

import styled from "styled-components";
import { RowFlex } from "./StyledComponents";

const Label = styled.label`
  display: inline-block;
  // border: 1px solid #ccc;

  border-radius: 25px;
  padding: 12.5px 12.5px;

  padding-left: 40px;
  padding-right: 40px;

  cursor: pointer;
`;

const BASE_URL = process.env.REACT_APP_BACKEND;

export default function UploadComponent() {
  const [items, setItems] = useState([]);

  const [hover1, setHover1] = useState(false);
  const [value, setValue] = useState("");

  const [hover2, setHover2] = useState(false);
  const [hover3, setHover3] = useState(false);

  const [selectedFile, setSelectedFile] = useState(null);

  async function getItems() {
    const response = await fetch(BASE_URL + "datatypes", {
      method: "get",
    });
    const data = await response.json();
    setItems(data);
  }

  const onFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    console.log(event.target.files[0]);
  };

  const onFileSubmit = async () => {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    const resp = await response.text();

    console.log(resp);
  };

  useEffect(() => {
    getItems();
  }, []);

  return (
    <div style={{ display: "flex" }}>
      <div
        style={{
          border: "1px solid #ccc",
          width: "50%",
          textAlign: "center",
        }}
      >
        <Divv bottom="0px" style={{ padding: "12.5px 12.5px" }}>
          Select one of the existing datasets...
        </Divv>

        <form>
          <div>
            <FormControl
              sx={{ width: "50%", margin: "20px" }}
              variant="outlined"
            >
              <InputLabel id="itemSelect">Data type</InputLabel>
              <Select
                labelId="itemId"
                id="item"
                label="itemSelect"
                onChange={(event) => {
                  console.log("Now selected", event.target.value);

                  setValue(event.target.value);
                }}
              >
                <MenuItem key="0" value="" disabled>
                  Choose a method
                </MenuItem>
                {items.map((item) => {
                  return (
                    <MenuItem key={item.id} value={item.method}>
                      {item.method}
                    </MenuItem>
                  );
                })}
              </Select>
            </FormControl>
          </div>

          <Divv top="0px">
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
                if (value === "") {
                  console.log("You need to select a method first...");
                  return;
                }
                console.log("sending over method", value);
              }}
              onMouseEnter={() => {
                setHover1(true);
              }}
              onMouseLeave={() => {
                setHover1(false);
              }}
            >
              Upload data
            </Button>
          </Divv>
        </form>
      </div>

      <div
        style={{ border: "1px solid #ccc", width: "50%", textAlign: "center" }}
      >
        <Divv>
          <Label
            style={{
              display: "inline-block",
              background: hover2 === false ? "white" : "#F4BB44",
              color: hover2 === false ? "black" : "black",
              transition: "color 0.4s linear",
              transition: "background 0.4s linear",
            }}
            onMouseEnter={() => setHover2(true)}
            onMouseLeave={() => setHover2(false)}
          >
            <input
              style={{
                display: "none",
              }}
              type="file"
              onChange={(event) => onFileChange(event)}
            />
            ...or click here to upload a new file.
          </Label>
        </Divv>

        <Divv top="0px" size="22.5px">
          You have uploaded: {selectedFile ? selectedFile.name : ""}
        </Divv>

        <Divv size="22.5px">
          in case of one file - what would you like the percentage of train/test
          data to be?
        </Divv>
        <Divv size="22.5px">what is the label column called?</Divv>
        <Divv size="22.5px">what is non-anomaly value of the label?</Divv>
        <Divv size="22.5px">
          would it be fine to save this file to our database?
        </Divv>

        <Divv top="0px">
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
              if (selectedFile === null) {
                console.log("You need to select a file first...");
                return;
              }
              console.log("sending over file", selectedFile.name);

              onFileSubmit();
            }}
            onMouseEnter={() => {
              setHover3(true);
            }}
            onMouseLeave={() => {
              setHover3(false);
            }}
          >
            Upload file
          </Button>
        </Divv>
      </div>
    </div>
  );
}
