import { useEffect, useState } from "react";
import { Divv, TextFieldFlex, Label } from "./StyledComponents";
import Xarrow from "react-xarrows";

import {
  Button,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  TextField,
  Checkbox,
} from "@mui/material";

const BASE_URL = process.env.REACT_APP_BACKEND;

export default function UploadComponent() {
  const [items, setItems] = useState([]);

  const [hover1, setHover1] = useState(false);
  const [value, setValue] = useState("");

  const [hover2, setHover2] = useState(false);
  const [hover3, setHover3] = useState(false);

  const [selectedFile, setSelectedFile] = useState(null);

  const [showXarrow, setShowXarrow] = useState(true);

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
    setShowXarrow(false);
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
              Use this dataset
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
            ...or click here to upload a new{" "}
            <span id="upload-something-here-end">file</span>.
          </Label>
        </Divv>

        <Divv top="0px" size="22.5px">
          {selectedFile ? (
            "You have uploaded: " + selectedFile.name
          ) : (
            <span id="upload-something-here-start">
              You have uploaded:&nbsp;&nbsp;
            </span>
          )}
          <Xarrow
            showXarrow={showXarrow}
            start="upload-something-here-start"
            end="upload-something-here-end"
            startAnchor="right"
            endAnchor="bottom"
            curveness="2.5"
            color="#FF5733"
          ></Xarrow>
        </Divv>

        <TextFieldFlex>
          <Divv size="22.5px">
            in case of one file - what would you like the percentage of train
            data to be?
            <Divv bottom="0px">
              <TextField
                error={false}
                helperText={false ? "emptyTitleMessage" : ""}
                id="percentage-field"
                variant="outlined"
                label="Percentage of Train Data"
              />
            </Divv>
          </Divv>
        </TextFieldFlex>

        <TextFieldFlex style={{ marginTop: "10px" }}>
          <Divv size="22.5px">
            what is the label column called?
            <Divv bottom="0px">
              <TextField
                error={false}
                helperText={false ? "emptyTitleMessage" : ""}
                id="label-column-field"
                variant="outlined"
                label="Label Column"
              />
            </Divv>
          </Divv>
        </TextFieldFlex>

        <TextFieldFlex style={{ marginTop: "10px" }}>
          <Divv size="22.5px">
            what is normal (non-anomaly) value of the label?
            <Divv bottom="0px">
              <TextField
                error={false}
                helperText={false ? "emptyTitleMessage" : ""}
                id="non-anomaly-value-field"
                variant="outlined"
                label="Normal Field Value"
              />
            </Divv>
          </Divv>
        </TextFieldFlex>

        <TextFieldFlex style={{ marginTop: "10px" }}>
          <Checkbox
            onChange={() => {
              console.log("hi");
            }}
            id="passCheckBox"
            color="default"
          />
          <Divv size="22.5px">
            save my data for future uses
            <Divv bottom="0px"></Divv>
          </Divv>
        </TextFieldFlex>

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
