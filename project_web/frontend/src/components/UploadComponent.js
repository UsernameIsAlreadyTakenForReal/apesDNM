import { useEffect, useState } from "react";
import { Divv, TextFieldFlex, Label, RowFlex } from "./StyledComponents";
import Xarrow from "react-xarrows";

import {
  Button,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  TextField,
  Checkbox,
  FormControlLabel,
  FormGroup,
} from "@mui/material";

const BASE_URL = process.env.REACT_APP_BACKEND;

export default function UploadComponent() {
  const [items, setItems] = useState([]);

  const [hover1, setHover1] = useState(false);
  const [value, setValue] = useState("");

  const [hover2, setHover2] = useState(false);
  const [hover3, setHover3] = useState(false);

  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedFile2, setSelectedFile2] = useState(null);

  const [showXarrow, setShowXarrow] = useState(false);

  // errors
  const [fileSelError, setFileSelError] = useState(false);
  const [fileSelErrorMessage, setFileSelErrorMessage] = useState(false);

  const [percentageError, setPercentageError] = useState(false);
  const [percentageErrorMessage, setPercentageErrorMessage] = useState("");

  const [labelColumnError, setLabelColumnError] = useState(false);
  const [labelColumnErrorMessage, setLabelColumnErrorMessage] = useState("");

  const [normalLabelError, setNormalLabelError] = useState(false);
  const [normalLabelErrorMessage, setNormalLabelErrorMessage] = useState("");

  const [saveData, setSaveData] = useState(false);

  async function getItems() {
    const response = await fetch(BASE_URL + "datatypes", {
      method: "get",
    });
    const data = await response.json();
    setItems(data);
  }

  const onFileChange = (event) => {
    setSelectedFile(null);
    setSelectedFile2(null);

    // console.log(event.target.files.length);
    if (event.target.files.length > 2) {
      setFileSelError(true);
      setShowXarrow(true);
      setFileSelErrorMessage("two files max can be selected");
      return;
    }

    if (event.target.files[0]) setSelectedFile(event.target.files[0]);
    if (event.target.files[1]) setSelectedFile2(event.target.files[1]);

    setShowXarrow(false);
  };

  const onFileSubmit = async () => {
    setPercentageError(false);
    setLabelColumnError(false);
    setNormalLabelError(false);

    // file logic
    if (!selectedFile) {
      setFileSelError(true);
      setShowXarrow(true);
      setFileSelErrorMessage("upload a file first");
      return;
    }

    // percentage logic
    const percentage = document.getElementById("percentage-field").value;

    if (percentage.length === 0) {
      setPercentageError(true);
      setPercentageErrorMessage("percentage cannot be empty");
      return;
    }

    if (percentage < 0 || percentage > 1) {
      setPercentageError(true);
      setPercentageErrorMessage("percentage must be between 0-1");
      return;
    }

    // label column logic
    const labelColumn = document.getElementById("label-column-field").value;

    if (labelColumn === "") {
      setLabelColumnError(true);
      setLabelColumnErrorMessage("unlabeled sets are not supported");
      return;
    }

    // normal value logic
    const normalValue = document.getElementById("normal-value-field").value;

    if (normalValue === "") {
      setNormalLabelError(true);
      setNormalLabelErrorMessage("a normal label must be provided");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("percentage", percentage);
    formData.append("labelColumn", labelColumn);
    formData.append("normalValue", normalValue);
    formData.append("saveData", saveData);

    console.log({
      file: formData,
      percentage: percentage,
      labelColumn: labelColumn,
      normalValue: normalValue,
      saveData: saveData,
    });

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
              // color: hover2 === false ? "black" : "black",
              // transition: "color 0.4s linear",
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
              multiple
            />
            ...or click here to upload the input{" "}
            <span id="upload-something-here-end">file(s)</span>
          </Label>
        </Divv>

        {selectedFile ? (
          <Divv top="0px" size="22.5px">
            {selectedFile2
              ? "You have uploaded: " +
                selectedFile.name +
                ", " +
                selectedFile2.name
              : "You have uploaded: " + selectedFile.name}
          </Divv>
        ) : fileSelError ? (
          <div style={{ transition: "color 0.4s linear" }}>
            <Divv top="0px" size="22.5px">
              <span id="upload-something-here-start">
                {fileSelErrorMessage}&nbsp;&nbsp;
              </span>

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
          </div>
        ) : (
          <></>
        )}

        <TextFieldFlex>
          <Divv
            size="22.5px"
            color={selectedFile2 ? "lightgray" : "black"}
            style={{
              margin: "25px",
              width: "60%",
            }}
          >
            in case of one file - what would you like the percentage of train
            data to be?
          </Divv>
          <TextField
            style={{ margin: "25px", width: "40%" }}
            error={percentageError}
            helperText={percentageError ? percentageErrorMessage : ""}
            id="percentage-field"
            variant="outlined"
            label="percentage of train data"
            disabled={selectedFile2 ? true : false}
          />
        </TextFieldFlex>

        <TextFieldFlex style={{ marginTop: "10px" }}>
          <Divv size="22.5px" style={{ margin: "25px", width: "60%" }}>
            what is the label column called?
          </Divv>
          <TextField
            style={{ margin: "25px", width: "40%" }}
            error={labelColumnError}
            helperText={labelColumnError ? labelColumnErrorMessage : ""}
            id="label-column-field"
            variant="outlined"
            label="label column"
          />
        </TextFieldFlex>

        <TextFieldFlex style={{ marginTop: "10px" }}>
          <Divv size="22.5px" style={{ margin: "25px", width: "60%" }}>
            what is normal (non-anomaly) value of the label?
          </Divv>
          <TextField
            style={{ margin: "25px", width: "40%" }}
            error={normalLabelError}
            helperText={normalLabelError ? normalLabelErrorMessage : ""}
            id="normal-value-field"
            variant="outlined"
            label="normal field label"
          />
        </TextFieldFlex>

        <TextFieldFlex style={{ marginTop: "10px" }}>
          <Divv size="22.5px" style={{ margin: "25px", width: "60%" }}>
            {/* save my data for future uses */}
          </Divv>
          <FormControlLabel
            style={{ margin: "25px", width: "40%" }}
            control={
              <Checkbox
                id="save-data-checkbox"
                color="default"
                onChange={() => setSaveData(!saveData)}
              />
            }
            label="save my data for future uses"
          />
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
            onClick={() => onFileSubmit()}
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
