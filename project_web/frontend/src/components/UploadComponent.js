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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
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

  const [saveDataCheckbox, setSaveDataCheckbox] = useState(false);
  const [labeledRadioValue, setLabeledRadioValue] = useState("yes");
  const [isSupervisedCheckbox, setIsSupervisedCheckbox] = useState(false);
  const [separateTrainAndTestCheckbox, setSeparateTrainAndTestCheckbox] =
    useState(true);

  const [classes, setClasses] = useState([]);
  const [classesTextfields, setClassesTextfields] = useState([1]);
  const [classesTextfieldsError, setClassesTextfieldsError] = useState(false);
  const [classesTextfieldsErrorMessage, setClassesTextfieldsErrorMessage] =
    useState(false);

  const [normalClass, setNormalClass] = useState("");

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

  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogText, setDialogText] = useState("");

  const [showExistingMethod, setShowExistingMethod] = useState(false);
  const [showFileUploadMethod, setShowFileUploadMethod] = useState(false);

  const [showExistingMethodButtonHover, setShowExistingMethodButtonHover] =
    useState(false);
  const [showFileUploadMethodButtonHover, setShowFileUploadMethodButtonHover] =
    useState(false);

  const [goBackButtonHover, setGoBackButtonHover] = useState(false);

  function resetAllFormErrorsAndData() {
    setDatasetError(false);
    setFileSelectionError(false);
    setPercentageError(false);
    setLabelColumnError(false);
    setClassesTextfieldsError(false);
    setNormalClassError(false);

    // document.getElementById("percentage-field").value = "";
    // document.getElementById("label-column-field").value = "";
    // document.getElementById("normal-class-field").value = "";

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

  function handleClassChange() {
    let classes = [];
    classesTextfields.forEach((textfield) => {
      if (document.getElementById("class-textfield" + textfield).value) {
        classes.push(
          document.getElementById("class-textfield" + textfield).value
        );
      }
    });

    console.log(classes);

    setClasses(classes);
    return classes;
  }

  async function onFileUpload() {
    setFileUploadButtonHover(false);

    setPercentageError(false);
    setLabelColumnError(false);
    setClassesTextfieldsError(false);
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

    // classes logic
    let classes = handleClassChange();

    if (classes.length !== classesTextfields.length) {
      setClassesTextfieldsError(true);
      setClassesTextfieldsErrorMessage("add a value for each class textfield");
      return;
    }

    if (new Set(classes).size !== classes.length) {
      setClassesTextfieldsError(true);
      setClassesTextfieldsErrorMessage("remove duplicate classes");
      return;
    }

    // normal class
    if (normalClass === "") {
      setNormalClassError(true);
      setNormalClassErrorMessage("normal class must be selected");
      return;
    }

    let files = selectedFiles.map((file) => file.name);

    let data = {
      files: files,
      isLabeled: labeledRadioValue === "idk" ? "unknown" : labeledRadioValue,
      label: labeledRadioValue === "yes" ? labelColumn : "-1",
      isSupervised: isSupervisedCheckbox,
      separateTrainAndTest: separateTrainAndTestCheckbox,
      trainDataPercentage: separateTrainAndTestCheckbox
        ? trainDataPercentage
        : "-1",
      classes: classes,
      normalClass: normalClass,
    };

    setDialogText(JSON.stringify(data, null, "\t"));

    setDialogOpen(true);
    return;

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
    <>
      {showExistingMethod === false && showFileUploadMethod === false ? (
        <>
          <Divv
            style={{
              display: "flex",
              justifyContent: "center",
            }}
          >
            choose the type of solution you want
          </Divv>
          <div
            style={{
              display: "flex",
              justifyContent: "center",
            }}
          >
            <Divv>
              <Button
                style={{
                  background:
                    showExistingMethodButtonHover === false
                      ? "black"
                      : "orange",
                  color:
                    showExistingMethodButtonHover === false ? "white" : "black",
                  fontWeight: "bold",
                }}
                variant="contained"
                color="primary"
                size="large"
                onMouseEnter={() => {
                  setShowExistingMethodButtonHover(true);
                }}
                onMouseLeave={() => {
                  setShowExistingMethodButtonHover(false);
                }}
                onClick={() => {
                  setShowExistingMethod(true);
                  setShowExistingMethodButtonHover(false);
                }}
              >
                Use Existing Methods
              </Button>
            </Divv>
            <Divv>
              <Button
                style={{
                  background:
                    showFileUploadMethodButtonHover === false
                      ? "black"
                      : "orange",
                  color:
                    showFileUploadMethodButtonHover === false
                      ? "white"
                      : "black",
                  fontWeight: "bold",
                }}
                variant="contained"
                color="primary"
                size="large"
                onMouseEnter={() => {
                  setShowFileUploadMethodButtonHover(true);
                }}
                onMouseLeave={() => {
                  setShowFileUploadMethodButtonHover(false);
                }}
                onClick={() => {
                  setShowFileUploadMethod(true);
                  setShowFileUploadMethodButtonHover(false);
                }}
              >
                Upload a new dataset
              </Button>
            </Divv>
          </div>
        </>
      ) : showExistingMethod === true && showFileUploadMethod === false ? (
        <div
          style={{
            // border: "1px solid #ccc",
            // width: "45%",
            textAlign: "center",
          }}
        >
          <Divv
            style={{
              textAlign: "left",
            }}
          >
            <Button
              style={{
                background: goBackButtonHover === false ? "black" : "orange",
                color: goBackButtonHover === false ? "white" : "black",
                fontWeight: "bold",
              }}
              variant="contained"
              color="primary"
              size="large"
              onClick={() => {
                setShowExistingMethod(false);
                setShowFileUploadMethod(false);
                setGoBackButtonHover(false);
              }}
              onMouseEnter={() => {
                setGoBackButtonHover(true);
              }}
              onMouseLeave={() => {
                setGoBackButtonHover(false);
              }}
            >
              Go back
            </Button>
          </Divv>

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
                  color:
                    existingDatasetButtonHover === false ? "white" : "black",
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
      ) : (
        <div
          style={{
            border: "1px solid #ccc",
            // width: "55%",
            textAlign: "center",
          }}
        >
          <Divv
            style={{
              textAlign: "left",
            }}
          >
            <Button
              style={{
                background: goBackButtonHover === false ? "black" : "orange",
                color: goBackButtonHover === false ? "white" : "black",
                fontWeight: "bold",
              }}
              variant="contained"
              color="primary"
              size="large"
              onClick={() => {
                setShowExistingMethod(false);
                setShowFileUploadMethod(false);
                setGoBackButtonHover(false);
              }}
              onMouseEnter={() => {
                setGoBackButtonHover(true);
              }}
              onMouseLeave={() => {
                setGoBackButtonHover(false);
              }}
            >
              Go back
            </Button>
          </Divv>

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
              click here to upload the input file(s).
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

          <TextFieldFlex style={{ marginTop: "10px" }}>
            <FormControl
              style={{
                margin: "25px",
                width: "50%",
                display: "flex",
                alignItems: "center",
              }}
            >
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
                  label="unsure"
                />
                <br></br>
              </RadioGroup>
            </FormControl>

            <FormControlLabel
              style={{ margin: "25px", width: "25%" }}
              control={
                <Checkbox
                  checked={isSupervisedCheckbox}
                  id="save-data-checkbox"
                  color="default"
                  onChange={() =>
                    setIsSupervisedCheckbox(!isSupervisedCheckbox)
                  }
                />
              }
              label="should it be supervised?"
            />

            <FormControlLabel
              style={{ margin: "25px", width: "25%" }}
              control={
                <Checkbox
                  checked={separateTrainAndTestCheckbox}
                  id="save-data-checkbox"
                  color="default"
                  onChange={() =>
                    setSeparateTrainAndTestCheckbox(
                      !separateTrainAndTestCheckbox
                    )
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
                width: "60%",
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
                style={{ margin: "25px", width: "40%" }}
                disabled={separateTrainAndTestCheckbox === false}
                error={percentageError}
                helperText={percentageError ? percentageErrorMessage : ""}
                id="percentage-field"
                variant="outlined"
                label="percentage of train data"
              />
            </Tooltip>
          </TextFieldFlex>

          <TextFieldFlex style={{ marginTop: "10px" }}>
            <Divv
              size="22.5px"
              style={{ margin: "25px", width: "60%" }}
              color={labeledRadioValue === "yes" ? "black" : "lightgray"}
            >
              what is the label column called?
            </Divv>
            <TextField
              style={{ margin: "25px", width: "40%" }}
              disabled={labeledRadioValue !== "yes"}
              error={labelColumnError}
              helperText={labelColumnError ? labelColumnErrorMessage : ""}
              id="label-column-field"
              variant="outlined"
              label="label column"
            />
          </TextFieldFlex>

          {classesTextfields.map((textfield) => {
            return (
              <TextFieldFlex style={{ marginTop: "10px" }}>
                <Divv size="22.5px" style={{ margin: "25px", width: "60%" }}>
                  {textfield === 1 ? (
                    <>
                      <span
                        style={{ cursor: "pointer" }}
                        onClick={() => {
                          if (classesTextfields.length === 1) return;
                          setClassesTextfields(classesTextfields.slice(0, -1));
                          handleClassChange();
                        }}
                      >
                        ((-)){" "}
                      </span>
                      what are the classes?
                      <span
                        style={{ cursor: "pointer" }}
                        onClick={() => {
                          setClassesTextfields([
                            ...classesTextfields,
                            classesTextfields.length + 1,
                          ]);
                          handleClassChange();
                        }}
                      >
                        {" "}
                        ((+))
                      </span>
                    </>
                  ) : (
                    ""
                  )}
                </Divv>

                <TextField
                  style={{
                    width: "40%",
                    display: "flex",
                    margin: "25px",
                    marginTop: textfield === 1 ? "25px" : "0px",
                    marginBottom:
                      textfield === classesTextfields.length ? "25px" : "0px",
                  }}
                  error={classesTextfieldsError}
                  helperText={
                    classesTextfieldsError &&
                    textfield === classesTextfields.length
                      ? classesTextfieldsErrorMessage
                      : ""
                  }
                  id={"class-textfield" + textfield}
                  variant="outlined"
                  label={"class #" + textfield}
                  onChange={() => handleClassChange()}
                />
              </TextFieldFlex>
            );
          })}

          <TextFieldFlex style={{ marginTop: "10px" }}>
            <Divv size="22.5px" style={{ margin: "25px", width: "60%" }}>
              what is normal (non-anomaly) class value?
            </Divv>
            <FormControl
              sx={{ width: "40%", margin: "25px" }}
              variant="outlined"
              error={normalClassError}
            >
              <InputLabel id="data-type-select">class type</InputLabel>
              <Select
                label="data-type-select"
                onChange={(event) => {
                  setNormalClass(event.target.value);
                  console.log(event.target.value);
                }}
              >
                <MenuItem key="0" value="" disabled>
                  choose the normal class
                </MenuItem>
                {classes.map((item, index) => {
                  return (
                    <MenuItem
                      key={index}
                      value={item === "" ? "no value at " + (index + 1) : item}
                    >
                      {item === "" ? "no value at " + (index + 1) : item}
                    </MenuItem>
                  );
                })}
              </Select>
              <FormHelperText>
                {normalClassError ? normalClassErrorMessage : ""}
              </FormHelperText>
            </FormControl>
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
                background:
                  fileUploadButtonHover === false ? "black" : "orange",
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
      )}

      <div style={{ display: "flex" }}>
        <WrapperDiv
          style={{
            display: loading ? "" : "none",
          }}
        >
          <CircularProgress />
          <Divv>loading...</Divv>
        </WrapperDiv>

        <Dialog open={dialogOpen} maxWidth="xl" fullWidt={true}>
          <DialogTitle>{"Proceed with these parameters?"}</DialogTitle>
          <DialogContent>
            <DialogContentText>
              <pre>{dialogText}</pre>
            </DialogContentText>
          </DialogContent>
          <DialogActions>
            <Button
              onClick={() => {
                setDialogOpen(false);
              }}
            >
              Go back
            </Button>
            <Button onClick={() => {}} autoFocus>
              Confirm
            </Button>
          </DialogActions>
        </Dialog>
      </div>
    </>
  );
}
