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
  CircularProgress,
} from "@mui/material";

const BASE_URL = process.env.REACT_APP_BACKEND;

export default function UploadComponent() {
  const [existingDatasets, setExistingDatasets] = useState([]);
  const [existingDatasetButtonHover, setExistingDatasetButtonHover] =
    useState(false);
  const [selectedDataset, setSelectedDataset] = useState("");

  const [fileInputHover, setFileInputHover] = useState(false);
  const [fileUploadButtonHover, setFileUploadButtonHover] = useState(false);

  const [selectedFile1, setSelectedFile1] = useState(null);
  const [selectedFile2, setSelectedFile2] = useState(null);

  const [showXarrow, setShowXarrow] = useState(false);

  const [saveDataCheckbox, setSaveDataCheckbox] = useState(false);

  const [loading, setLoading] = useState(false);

  // --------------------------------- errors --------------------------------
  const [fileSelectionError, setFileSelectionError] = useState(false);
  const [fileSelectionErrorMessage, setFileSelectionErrorMessage] =
    useState(false);

  const [percentageError, setPercentageError] = useState(false);
  const [percentageErrorMessage, setPercentageErrorMessage] = useState("");

  const [labelColumnError, setLabelColumnError] = useState(false);
  const [labelColumnErrorMessage, setLabelColumnErrorMessage] = useState("");

  const [normalLabelError, setNormalLabelError] = useState(false);
  const [normalLabelErrorMessage, setNormalLabelErrorMessage] = useState("");

  function resetAllFormErrors() {
    setFileSelectionError(false);
    setPercentageError(false);
    setLabelColumnError(false);
    setNormalLabelError(false);
  }

  // -------------------------------------------------------------------------

  async function getExistingDatasetItems() {
    const response = await fetch(BASE_URL + "datasets", {
      method: "get",
    });
    const data = await response.json();
    setExistingDatasets(data);
  }

  async function onFileChange(event) {
    setSelectedFile1(null);
    setSelectedFile2(null);

    resetAllFormErrors();

    if (event.target.files.length > 2) {
      setShowXarrow(true);
      setFileSelectionError(true);
      setFileSelectionErrorMessage("two files max can be selected...");

      setSelectedFile1(null);
      setSelectedFile2(null);
      return;
    }

    if (event.target.files[0]) setSelectedFile1(event.target.files[0]);
    if (event.target.files[1]) setSelectedFile2(event.target.files[1]);

    if (
      event.target.files[0].name.includes(".zip") ||
      event.target.files[0].name.includes(".rar") ||
      event.target.files[1].name.includes(".zip") ||
      event.target.files[1].name.includes(".rar")
    ) {
      if (event.target.files[0] && event.target.files[1]) {
        setFileSelectionError(true);
        setFileSelectionErrorMessage(
          "if sending archives, send only one file..."
        );
        setShowXarrow(true);

        setSelectedFile1(null);
        setSelectedFile2(null);
        return;
      }

      const formData = new FormData();
      formData.append("file", event.target.files[0]);

      // find out if we have one or two files in the archive

      // const response = await fetch("/unarchive", {
      //   method: "POST",
      //   body: formData,
      // });

      // const resp = await response.text();
      // console.log(resp);
    }

    setShowXarrow(false);
  }

  const onFileSubmit = async () => {
    setPercentageError(false);
    setLabelColumnError(false);
    setNormalLabelError(false);

    // ----------------------------- file logic ------------------------------
    if (!selectedFile1) {
      setShowXarrow(true);
      setFileSelectionError(true);
      setFileSelectionErrorMessage("upload a file first...");
      return;
    }

    // --------------------------- percentage logic --------------------------
    const trainDataPercentage =
      document.getElementById("percentage-field").value;

    if (!selectedFile2) {
      if (trainDataPercentage.length === 0) {
        setPercentageError(true);
        setPercentageErrorMessage("percentage cannot be empty...");
        return;
      }

      if (trainDataPercentage < 0 || trainDataPercentage > 1) {
        setPercentageError(true);
        setPercentageErrorMessage("percentage must be between 0-1...");
        return;
      }
    }

    // ------------------------- label column logic --------------------------
    const labelColumn = document.getElementById("label-column-field").value;

    if (labelColumn === "") {
      setLabelColumnError(true);
      setLabelColumnErrorMessage("unlabeled sets are not supported...");
      return;
    }

    // ------------------------- normal value logic --------------------------
    const normalLabel = document.getElementById("normal-value-field").value;

    if (normalLabel === "") {
      setNormalLabelError(true);
      setNormalLabelErrorMessage("a normal label value must be provided...");
      return;
    }

    const formData = new FormData();

    formData.append("file1", selectedFile1);
    if (selectedFile2) {
      formData.append("file2", selectedFile2);
    }

    formData.append("percentage", trainDataPercentage);
    formData.append("labelColumn", labelColumn);
    formData.append("normalLabel", normalLabel);
    formData.append("saveData", saveDataCheckbox);

    // console.log({
    //   file: formData,
    //   percentage: trainDataPercentage,
    //   labelColumn: labelColumn,
    //   normalLabel: normalLabel,
    //   saveData: saveDataCheckbox,
    // });

    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    const resp = await response.text();
    console.log(resp);
  };

  useEffect(() => {
    getExistingDatasetItems();
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

                  setSelectedDataset(event.target.value);
                }}
              >
                <MenuItem key="0" value="" disabled>
                  Choose a method
                </MenuItem>
                {existingDatasets.map((item) => {
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
                background:
                  existingDatasetButtonHover === false ? "black" : "orange",
                color: existingDatasetButtonHover === false ? "white" : "black",
                fontWeight: "bold",
              }}
              variant="contained"
              color="primary"
              size="large"
              onClick={() => {
                if (selectedDataset === "") {
                  console.log("You need to select a method first...");
                  return;
                }
                console.log("sending over method", selectedDataset);
              }}
              onMouseEnter={() => {
                setExistingDatasetButtonHover(true);
              }}
              onMouseLeave={() => {
                setExistingDatasetButtonHover(false);
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
              background: fileInputHover === false ? "white" : "#F4BB44",
              transition: "background 0.4s linear",
            }}
            onMouseEnter={() => setFileInputHover(true)}
            onMouseLeave={() => setFileInputHover(false)}
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

        {selectedFile1 && !fileSelectionError ? (
          <Divv top="0px" size="22.5px">
            {selectedFile2
              ? "You have uploaded: " +
                selectedFile1.name +
                ", " +
                selectedFile2.name
              : "You have uploaded: " + selectedFile1.name}
          </Divv>
        ) : fileSelectionError ? (
          <div style={{ transition: "color 0.4s linear" }}>
            <Divv top="0px" size="22.5px" style={{ color: "red" }}>
              <span id="upload-something-here-start">
                {fileSelectionErrorMessage}&nbsp;&nbsp;
              </span>

              <Xarrow
                showXarrow={false}
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
            color={selectedFile2 && !fileSelectionError ? "lightgray" : "black"}
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
            disabled={selectedFile2 && !fileSelectionError ? true : false}
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
                onChange={() => setSaveDataCheckbox(!saveDataCheckbox)}
              />
            }
            label="save my data for future uses"
          />
        </TextFieldFlex>

        <Divv>
          <Button
            style={{
              background: fileUploadButtonHover === false ? "black" : "orange",
              color: fileUploadButtonHover === false ? "white" : "black",
              fontWeight: "bold",
            }}
            variant="contained"
            color="primary"
            size="large"
            onClick={() => onFileSubmit()}
            onMouseEnter={() => {
              setFileUploadButtonHover(true);
            }}
            onMouseLeave={() => {
              setFileUploadButtonHover(false);
            }}
          >
            Upload file
          </Button>
        </Divv>
      </div>
    </div>
  );
}
