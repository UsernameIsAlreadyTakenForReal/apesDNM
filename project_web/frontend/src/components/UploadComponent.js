import { useEffect, useState } from "react";
import * as React from "react";
import { Divv, TextFieldFlex, Label, WrapperDiv } from "./StyledComponents";
import { tsParticles } from "tsparticles-engine";
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
  Tooltip,
  Typography,
  FormHelperText,
} from "@mui/material";

const BASE_URL = process.env.REACT_APP_BACKEND;

export default function UploadComponent() {
  const [existingDatasets, setExistingDatasets] = useState([]);
  const [existingDatasetButtonHover, setExistingDatasetButtonHover] =
    useState(false);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedMethods, setSelectedMethods] = useState([]);

  const [fileInputHover, setFileInputHover] = useState(false);
  const [fileUploadButtonHover, setFileUploadButtonHover] = useState(false);

  const [selectedFiles, setSelectedFiles] = useState([]);

  const [showXarrow, setShowXarrow] = useState(false);

  const [saveDataCheckbox, setSaveDataCheckbox] = useState(false);

  const [loading, setLoading] = useState(false);

  // errors
  const [fileSelectionError, setFileSelectionError] = useState(false);
  const [fileSelectionErrorMessage, setFileSelectionErrorMessage] =
    useState(false);

  const [datasetError, setDatasetError] = useState(false);
  const [datasetErrorMessage, setDatasetErrorMessage] = useState(false);

  const [percentageError, setPercentageError] = useState(false);
  const [percentageErrorMessage, setPercentageErrorMessage] = useState("");

  const [labelColumnError, setLabelColumnError] = useState(false);
  const [labelColumnErrorMessage, setLabelColumnErrorMessage] = useState("");

  const [normalLabelError, setNormalLabelError] = useState(false);
  const [normalLabelErrorMessage, setNormalLabelErrorMessage] = useState("");

  const [stringOfFilesUploaded, setStringOfFilesUploaded] = useState("");

  function resetAllFormErrorsAndData() {
    setDatasetError(false);

    setFileSelectionError(false);
    setPercentageError(false);
    setLabelColumnError(false);
    setNormalLabelError(false);

    document.getElementById("percentage-field").value = "";
    document.getElementById("label-column-field").value = "";
    document.getElementById("normal-value-field").value = "";
    setSaveDataCheckbox(false);
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
    resetAllFormErrorsAndData();
    setSelectedFiles(event.target.files);

    let archiveFoundAndMultipleFiles = false;

    if (event.target.files.length > 1) {
      [...event.target.files].forEach((file) => {
        if (file.name.includes(".zip") || file.name.includes(".rar")) {
          archiveFoundAndMultipleFiles = true;
        }
      });
    }

    if (archiveFoundAndMultipleFiles) {
      setFileSelectionError(true);
      setFileSelectionErrorMessage("if sending archives, send only one...");
      setSelectedFiles([]);
      return;
    }

    let localStringOfFilesUploaded = "";

    [...event.target.files].forEach((file) => {
      localStringOfFilesUploaded =
        localStringOfFilesUploaded + file.name + ", ";
    });

    localStringOfFilesUploaded = localStringOfFilesUploaded.slice(0, -2);
    setStringOfFilesUploaded(localStringOfFilesUploaded);
  }

  async function onFileUpload() {
    setFileUploadButtonHover(false);
    setPercentageError(false);
    setLabelColumnError(false);
    setNormalLabelError(false);

    // file logic
    if (!selectedFiles[0]) {
      setShowXarrow(true);
      setFileSelectionError(true);
      setFileSelectionErrorMessage("upload a file first...");
      return;
    }

    // percentage logic
    let trainDataPercentage =
      document.getElementById("percentage-field").value !== ""
        ? Number(document.getElementById("percentage-field").value)
        : 0.7;

    document.getElementById("percentage-field").value = trainDataPercentage;

    console.log(trainDataPercentage);

    if (isNaN(trainDataPercentage)) {
      setPercentageError(true);
      setPercentageErrorMessage("percentage must be a number (between 0-1)");
      return;
    }

    if (trainDataPercentage < 0 || trainDataPercentage > 1) {
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
    const normalLabel = document.getElementById("normal-value-field").value;

    if (normalLabel === "") {
      setNormalLabelError(true);
      setNormalLabelErrorMessage("a normal label value must be provided");
      return;
    }

    const formData = new FormData();

    formData.append("file0", selectedFiles[0]);
    if (selectedFiles.length === 2) {
      formData.append("file1", selectedFiles[1]);
    }

    formData.append("percentage", trainDataPercentage);
    formData.append("labelColumn", labelColumn);
    formData.append("normalLabel", normalLabel);
    formData.append("saveData", saveDataCheckbox);

    console.log("body is", {
      file: formData,
      percentage: trainDataPercentage,
      labelColumn: labelColumn,
      normalLabel: normalLabel,
      saveData: saveDataCheckbox,
    });

    setLoading(true);

    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    const textResponse = await response.text();
    console.log(textResponse);

    setLoading(false);
  }

  function handleEKGMethodsCheckboxes(methodNumber) {
    let tempSelectedMethods = selectedMethods;

    if (tempSelectedMethods.includes(methodNumber)) {
      tempSelectedMethods.splice(tempSelectedMethods.indexOf(methodNumber), 1);
    } else tempSelectedMethods.push(methodNumber);

    setSelectedMethods(tempSelectedMethods.sort());
  }

  async function onUseThisDataset() {
    setDatasetError(false);
    setDatasetErrorMessage("");

    if (selectedDataset === "") {
      setDatasetError(true);
      setDatasetErrorMessage("you need to select a data-set first...");
      return;
    }

    if (selectedDataset === "EKG" && selectedMethods.length === 0) {
      setDatasetError(true);
      setDatasetErrorMessage("you need to select a method for EKG first...");
      return;
    }

    if (selectedDataset !== "EKG") {
      handleEKGMethodsCheckboxes(1);
    }

    const formData = new FormData();

    formData.append("dataset", selectedDataset);
    formData.append("methods", selectedMethods);

    console.log(
      "dataset is",
      selectedDataset,
      "selected methods are",
      selectedMethods
    );

    setLoading(true);

    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    const resp = await response.text();

    setLoading(false);
    console.log(resp);
  }

  useEffect(() => {
    getExistingDatasetItems();
  }, []);

  return (
    <div style={{ display: "flex" }}>
      <div
        style={{
          border: "1px solid #ccc",
          width: "45%",
          textAlign: "center",
        }}
      >
        <Divv bottom="0px" style={{ padding: "12.5px 12.5px" }}>
          select one of the existing datasets...
        </Divv>

        <form>
          <div>
            <FormControl
              sx={{ width: "50%", margin: "20px" }}
              variant="outlined"
              error={datasetError}
            >
              <InputLabel id="data-type-select">data type</InputLabel>
              <Select
                label="data-type-select"
                onChange={(event) => {
                  console.log("Now selected", event.target.value);

                  setSelectedDataset(event.target.value);
                  setSelectedMethods([]);
                }}
              >
                <MenuItem key="0" value="" disabled>
                  choose a dataset
                </MenuItem>
                {existingDatasets.map((item) => {
                  return (
                    <MenuItem key={item.id} value={item.dataset}>
                      {item.dataset}
                    </MenuItem>
                  );
                })}
              </Select>
              <FormHelperText>
                {datasetError ? datasetErrorMessage : ""}
              </FormHelperText>
            </FormControl>
          </div>

          {selectedDataset === "EKG" ? (
            <div style={{ marginBottom: "20px" }}>
              <FormControlLabel
                control={
                  <Checkbox
                    color="default"
                    onChange={() => {
                      handleEKGMethodsCheckboxes(1);
                    }}
                  />
                }
                label="method #1"
              />
              <FormControlLabel
                control={
                  <Checkbox
                    color="default"
                    onChange={() => {
                      handleEKGMethodsCheckboxes(2);
                    }}
                  />
                }
                label="method #2"
              />
              <FormControlLabel
                control={
                  <Checkbox
                    color="default"
                    onChange={() => {
                      handleEKGMethodsCheckboxes(3);
                    }}
                  />
                }
                label="method #3"
              />
            </div>
          ) : (
            <></>
          )}

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
              onClick={() => onUseThisDataset()}
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
        style={{
          border: "1px solid #ccc",
          width: "55%",
          textAlign: "center",
        }}
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
            ...or click here to upload the input file(s)
          </Label>
        </Divv>

        {selectedFiles[0] && !fileSelectionError ? (
          <Divv top="0px" size="22.5px">
            you have uploaded: {stringOfFilesUploaded}
          </Divv>
        ) : (
          <div
            style={{
              display: fileSelectionError ? "" : "none",
              transition: "color 0.4s linear",
            }}
          >
            <Divv top="0px" size="22.5px" style={{ color: "red" }}>
              {fileSelectionErrorMessage}&nbsp;&nbsp;
            </Divv>
          </div>
        )}

        <TextFieldFlex>
          <Divv
            size="22.5px"
            style={{
              margin: "25px",
              width: "60%",
            }}
          >
            what would you like the percentage of train data to be?
          </Divv>

          <Tooltip
            title={<Typography fontSize={14}>default value is 0.7</Typography>}
            placement="top"
            arrow={false}
          >
            <TextField
              style={{ margin: "25px", width: "40%" }}
              error={percentageError}
              helperText={percentageError ? percentageErrorMessage : ""}
              id="percentage-field"
              variant="outlined"
              label="percentage of train data"
              InputLabelProps={{
                shrink:
                  document.getElementById("percentage-field").value !== "",
              }}
            />
          </Tooltip>
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
                checked={saveDataCheckbox}
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
            disabled={loading === true}
            variant="contained"
            color="primary"
            size="large"
            onClick={() => onFileUpload()}
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

      <WrapperDiv
        style={{
          display: loading ? "" : "none",
        }}
      >
        <CircularProgress />
        <Divv>loading...</Divv>
      </WrapperDiv>
    </div>
  );
}
