import { useEffect, useState } from "react";
import * as React from "react";
import { Divv, TextFieldFlex, Label, WrapperDiv } from "./StyledComponents";

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
  FormLabel,
  RadioGroup,
  Radio,
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

  const [datasetClasses, setDatasetClasses] = useState([]);
  const [datasetNormalClass, setDatasetNormalClass] = useState("");

  const [saveDataCheckbox, setSaveDataCheckbox] = useState(false);
  const [labeledRadioValue, setLabeledRadioValue] = useState("yes");
  const [isSupervisedCheckbox, setIsSupervisedCheckbox] = useState(false);
  const [separateTrainAndTestCheckbox, setSeparateTrainAndTestCheckbox] =
    useState(true);

  const [classesTextfields, setClassesTextfields] = useState([1]);

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

  const [normalClassError, setNormalClassError] = useState(false);
  const [normalClassErrorMessage, setNormalClassErrorMessage] = useState("");

  const [stringOfFilesUploaded, setStringOfFilesUploaded] = useState("");

  function resetAllFormErrorsAndData() {
    setDatasetError(false);

    setFileSelectionError(false);
    setPercentageError(false);
    setLabelColumnError(false);
    setNormalClassError(false);

    document.getElementById("percentage-field").value = "";
    document.getElementById("label-column-field").value = "";
    document.getElementById("normal-class-value-field").value = "";

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
    console.log([...event.target.files]);

    resetAllFormErrorsAndData();
    setSelectedFiles([...event.target.files]);

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
    setNormalClassError(false);

    // file logic
    if (!selectedFiles[0]) {
      setFileSelectionError(true);
      setFileSelectionErrorMessage("upload a file first...");
      return;
    }

    // percentage logic
    let trainDataPercentage = "";
    if (separateTrainAndTestCheckbox) {
      trainDataPercentage =
        document.getElementById("percentage-field").value !== ""
          ? Number(document.getElementById("percentage-field").value)
          : 0.7;

      document.getElementById("percentage-field").value = trainDataPercentage;

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
    }
    // label column logic
    let labelColumn = "";
    if (labeledRadioValue === "yes") {
      labelColumn = document.getElementById("label-column-field").value;

      if (labelColumn === "") {
        setLabelColumnError(true);
        setLabelColumnErrorMessage(
          "labeled dataset selected. provide a target"
        );
        return;
      }
    }

    let classes = [];

    classesTextfields.forEach((textfield) => {
      if (document.getElementById("classTextfield" + textfield).value) {
        classes.push(
          document.getElementById("classTextfield" + textfield).value
        );
      }
    });

    console.log(classes);

    // normal value logic
    const normalClass = document.getElementById(
      "normal-class-value-field"
    ).value;

    if (normalClass === "") {
      setNormalClassError(true);
      setNormalClassErrorMessage("a normal class value must be provided");
      return;
    }

    if (classes.includes(normalClass) === false) {
      setNormalClassError(true);
      setNormalClassErrorMessage("normal class must be among classes");
      return;
    }

    const formData = new FormData();

    for (let i = 0; i < selectedFiles.length; i++) {
      formData.append(`file${i}`, selectedFiles[i]);
    }

    formData.append("percentage", trainDataPercentage);
    formData.append("labelColumn", labelColumn);
    formData.append("normalClass", normalClass);
    formData.append("saveData", saveDataCheckbox);

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
            you have uploaded{" "}
            {selectedFiles.length === 1
              ? "1 file"
              : selectedFiles.length + " files"}
            : <br></br> {stringOfFilesUploaded}
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

        <TextFieldFlex style={{ marginTop: "10px" }} flexDirection="">
          <FormControl style={{ margin: "25px", width: "35%" }}>
            <FormLabel>is the dataset labeled?</FormLabel>
            <RadioGroup
              row
              name="row-radio-buttons-group"
              defaultValue="yes"
              value={labeledRadioValue}
              onChange={(event) => {
                setLabeledRadioValue(event.target.value);
              }}
            >
              <FormControlLabel value="yes" control={<Radio />} label="yes" />
              <FormControlLabel value="no" control={<Radio />} label="no" />
              <FormControlLabel
                value="idk"
                control={<Radio />}
                label="idk man..."
              />
              <br></br>
            </RadioGroup>
          </FormControl>

          <FormControlLabel
            style={{ margin: "25px", width: "30%" }}
            control={
              <Checkbox
                checked={isSupervisedCheckbox}
                id="save-data-checkbox"
                color="default"
                onChange={() => setIsSupervisedCheckbox(!isSupervisedCheckbox)}
              />
            }
            label="should it be supervised?"
          />

          <FormControlLabel
            style={{ margin: "25px", width: "35%" }}
            control={
              <Checkbox
                checked={separateTrainAndTestCheckbox}
                id="save-data-checkbox"
                color="default"
                onChange={() =>
                  setSeparateTrainAndTestCheckbox(!separateTrainAndTestCheckbox)
                }
              />
            }
            label="separate train and test data?"
          />
        </TextFieldFlex>

        <TextFieldFlex>
          <Divv
            size="22.5px"
            style={{
              margin: "25px",
              width: "55%",
            }}
            color={separateTrainAndTestCheckbox ? "black" : "lightgray"}
          >
            what would you like the percentage of train data to be?
          </Divv>

          <Tooltip
            title={
              separateTrainAndTestCheckbox ? (
                <Typography fontSize={14}>default value is 0.7</Typography>
              ) : (
                ""
              )
            }
            placement="top"
            arrow={false}
          >
            <TextField
              style={{ margin: "25px", width: "45%" }}
              disabled={separateTrainAndTestCheckbox === false}
              error={percentageError}
              helperText={percentageError ? percentageErrorMessage : ""}
              id="percentage-field"
              variant="outlined"
              label="percentage of train data"
              InputLabelProps={{
                shrink: true,
              }}
            />
          </Tooltip>
        </TextFieldFlex>

        <TextFieldFlex style={{ marginTop: "10px" }}>
          <Divv
            size="22.5px"
            style={{ margin: "25px", width: "55%" }}
            color={labeledRadioValue === "yes" ? "black" : "lightgray"}
          >
            what is the label column called?
          </Divv>
          <TextField
            style={{ margin: "25px", width: "45%" }}
            disabled={labeledRadioValue !== "yes"}
            error={labelColumnError}
            helperText={labelColumnError ? labelColumnErrorMessage : ""}
            id="label-column-field"
            variant="outlined"
            label="label column"
            InputLabelProps={{
              shrink: true,
            }}
          />
        </TextFieldFlex>

        {classesTextfields.map((textfield) => {
          return (
            <TextFieldFlex style={{ marginTop: "10px" }}>
              <Divv
                size="22.5px"
                style={{ margin: "25px", width: "55%" }}
                color={labeledRadioValue === "yes" ? "black" : "lightgray"}
              >
                {textfield === 1 ? "what are the classes?" : ""}
              </Divv>
              <span
                style={{
                  width: "45%",
                  margin: "25px",
                  display: "flex",
                  justifyContent: "center",
                }}
              >
                <TextField
                  style={{
                    width: [
                      classesTextfields.length - 1,
                      classesTextfields.length,
                    ].includes(textfield)
                      ? "70%"
                      : "100%",
                    flexDirection: "column",
                  }}
                  disabled={labeledRadioValue !== "yes"}
                  helperText={labelColumnError ? labelColumnErrorMessage : ""}
                  id={"classTextfield" + textfield}
                  variant="outlined"
                  label={"class #" + textfield}
                  InputLabelProps={{
                    shrink: true,
                  }}
                />
                {[
                  classesTextfields.length - 1,
                  classesTextfields.length,
                ].includes(textfield) ? (
                  <Button
                    style={{
                      width: "30%",
                      background: "black",
                      color: "white",
                      fontWeight: "bold",
                    }}
                    variant="contained"
                    color="primary"
                    size="large"
                    onClick={() => {
                      if (textfield === classesTextfields.length)
                        setClassesTextfields([
                          ...classesTextfields,
                          classesTextfields.length + 1,
                        ]);
                      else setClassesTextfields(classesTextfields.slice(0, -1));
                    }}
                  >
                    {textfield === classesTextfields.length - 1 ? "-" : "+"}
                  </Button>
                ) : (
                  ""
                )}{" "}
              </span>
            </TextFieldFlex>
          );
        })}

        <TextFieldFlex style={{ marginTop: "10px" }}>
          <Divv size="22.5px" style={{ margin: "25px", width: "55%" }}>
            what is normal (non-anomaly) class value?
          </Divv>
          <TextField
            style={{ margin: "25px", width: "45%" }}
            error={normalClassError}
            helperText={normalClassError ? normalClassErrorMessage : ""}
            id="normal-class-value-field"
            variant="outlined"
            label="normal class value"
            InputLabelProps={{
              shrink: true,
            }}
          />
        </TextFieldFlex>

        <TextFieldFlex style={{ marginTop: "10px" }}>
          <Divv size="22.5px" style={{ margin: "25px", width: "55%" }}>
            {/* save my data for future uses */}
          </Divv>
          <FormControlLabel
            style={{ margin: "25px", width: "45%" }}
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
