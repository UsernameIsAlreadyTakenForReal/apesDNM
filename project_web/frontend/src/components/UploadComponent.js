import { useEffect, useState } from "react";
import { Divv, RowFlex } from "./StyledComponents";
import {
  Button,
  FormControl,
  FormHelperText,
  InputLabel,
  MenuItem,
  Select,
} from "@mui/material";

import { Input } from "@mui/material";
import styled from "styled-components";

const Labell = styled.label`
  display: inline-block;
  // border: 1px solid #ccc;

  border-radius: 15px;
  padding: 15px 15px;
  cursor: pointer;
`;
const Inputt = styled.input`
  display: none;
`;

const BASE_URL = process.env.REACT_APP_BACKEND;
console.log("url is", BASE_URL);

export default function UploadComponent() {
  const [items, setItems] = useState([]);
  const [singularItem, setSingularItem] = useState([]);

  const [hover1, setHover1] = useState(false);
  const [value, setValue] = useState("");

  const [hover2, setHover2] = useState(false);

  const [selectedFile, setSelectedFile] = useState(null);

  async function getItems() {
    const response = await fetch(BASE_URL + "datatypes", {
      method: "get",
    });
    const data = await response.json();
    setItems(data);
  }

  async function getSingularItem() {
    if (value === "") return;
    console.log(BASE_URL + "datatype?id=" + value);
    const response = await fetch(BASE_URL + "datatype?id=" + value, {
      method: "post",
    });
    const data = await response.text();
    setSingularItem(data);
  }
  const handleFileSelect = (event) => {
    setSelectedFile(event.target.files[0]);
    console.log(event.target.files[0]);
  };

  const handleSubmit = async () => {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });
  };

  useEffect(() => {
    getItems();
  }, []);

  return (
    <>
      <Divv>Select one of the existing types of data.</Divv>

      <form>
        <div style={{ flexDirection: "horizontal" }}>
          <FormControl sx={{ width: "30%", margin: "20px" }} variant="outlined">
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
                  <MenuItem key={item.id} value={item.id}>
                    {item.method}
                  </MenuItem>
                );
              })}
            </Select>
            {/* {value === "" ? (
            <FormHelperText>
              Please select a data type from the ones previously processed
            </FormHelperText>
          ) : (
            <></>
          )} */}
          </FormControl>
          <Divv top="0px">
            <Labell
              class="custom-file-upload"
              style={{
                display: "inline-block",
                background: hover2 === false ? "white" : "grey",
                color: hover2 === false ? "black" : "white",
                transition: "color 0.4s linear",
                transition: "background 0.4s linear",
              }}
              onMouseEnter={() => setHover2(true)}
              onMouseLeave={() => setHover2(false)}
            >
              <Inputt
                type="file"
                onChange={(event) => handleFileSelect(event)}
              />
              Click here to upload the file...
            </Labell>
          </Divv>
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
              if (selectedFile === null) {
                console.log("You need to select a file first...");
                return;
              }
              console.log("sending over file", selectedFile.name);

              handleSubmit();
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
    </>
  );
}
