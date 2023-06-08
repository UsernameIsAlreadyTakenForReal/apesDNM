import { useEffect, useState } from "react";

import { Divv, TextFieldFlex, Label } from "./StyledComponents";
import WebSocketComponent from "./WebSocketComponent";

import lstmSVG from "../lstm.svg";
import cnnSVG from "../cnn.svg";

import Terminal from "react-terminal-ui";

import ImageViewer from "react-simple-image-viewer";

import {
  Button,
  Backdrop,
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
  Slider,
} from "@mui/material";

const BASE_URL = process.env.REACT_APP_BACKEND;

export default function UploadComponent() {
  // show/hide elements
  const [showExistingMethod, setShowExistingMethod] = useState(false);
  const [showFileUploadMethod, setShowFileUploadMethod] = useState(false);

  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogText, setDialogText] = useState("");

  const [loading, setLoading] = useState(false);
  const [loadingText, setLoadingText] = useState("");

  const [showResults, setShowResults] = useState(false);

  // data
  const [existingDatasets, setExistingDatasets] = useState([]);

  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedMethods, setSelectedMethods] = useState([]);

  const [selectedFiles, setSelectedFiles] = useState([]);

  const [labeledRadioValue, setLabeledRadioValue] = useState("yes");

  const [isSupervisedCheckbox, setIsSupervisedCheckbox] = useState(false);
  const [shuffleRows, setShuffleRows] = useState(true);
  const [separateTrainAndTestCheckbox, setSeparateTrainAndTestCheckbox] =
    useState(true);
  const [saveDataCheckbox, setSaveDataCheckbox] = useState(false);

  const [trainDataPercentage, setTrainDataPercentage] = useState("");
  const [labelColumn, setLabelColumn] = useState("");

  const [classesTextfields, setClassesTextfields] = useState([1, 2]);
  const [classes, setClasses] = useState([]);

  const [normalClass, setNormalClass] = useState("");

  const [epochs, setEpochs] = useState(40);

  const [responseData, setResponseData] = useState(null);

  const [backendMLPlots, setBackendMLPlots] = useState([]);
  const [backendCptions, setBackendCptions] = useState([]);
  const [backendConsole, setBackendConsole] = useState([]);
  const [backendResults, setBackendResults] = useState("");

  // hovers
  const [existingDatasetButtonHover, setExistingDatasetButtonHover] =
    useState(false);
  const [fileUploadButtonHover, setFileUploadButtonHover] = useState(false);

  const [showExistingMethodButtonHover, setShowExistingMethodButtonHover] =
    useState(false);
  const [showFileUploadMethodButtonHover, setShowFileUploadMethodButtonHover] =
    useState(false);

  const [fileInputHover, setFileInputHover] = useState(false);

  const [goBackButtonHover, setGoBackButtonHover] = useState(false);

  const [addClassButtonHover, setAddClassButtonHover] = useState(false);
  const [removeClassButtonHover, setRemoveClassButtonHover] = useState(false);

  const [dialogBackButtonHover, setDialogBackButtonHover] = useState(false);
  const [dialogConfirmButtonHover, setDialogConfirmButtonHover] =
    useState(false);

  const [method1Hover, setMethod1Hover] = useState(false);
  const [method2Hover, setMethod2Hover] = useState(false);
  const [method3Hover, setMethod3Hover] = useState(false);

  const [supervisedCheckboxHover, setSupervisedCheckboxHover] = useState(false);

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

  const [classesTextfieldsError, setClassesTextfieldsError] = useState(false);
  const [classesTextfieldsErrorMessage, setClassesTextfieldsErrorMessage] =
    useState(false);

  // misc
  const [stringOfFilesUploaded, setStringOfFilesUploaded] = useState("");

  // image viewer
  const [imageViewerOpen, setImageViewerOpen] = useState(false);
  const [currentImage, setCurrentImage] = useState(0);

  // -------------------------------------------------------------------------
  const delay = (ms) => new Promise((res) => setTimeout(res, ms));

  // -------------------------------------------------------------------------
  function resetAllFormErrorsAndData() {
    setSelectedDataset("");
    setSelectedMethods([]);

    setSelectedFiles([]);

    setLabeledRadioValue("yes");
    setIsSupervisedCheckbox(false);
    setSeparateTrainAndTestCheckbox(true);
    setShuffleRows(true);

    setTrainDataPercentage("");

    setLabelColumn("");

    setClassesTextfields([1, 2]);
    setClasses([]);
    setNormalClass("");

    setEpochs(40);

    setSaveDataCheckbox(false);

    setFileSelectionError(false);
    setDatasetError(false);
    setPercentageError(false);
    setLabelColumnError(false);
    setNormalClassError(false);
    setClassesTextfieldsError(false);
  }

  // -------------------------------------------------------------------------
  async function getExistingDatasetItems() {
    const response = await fetch(BASE_URL + "datasets", {
      method: "get",
    });
    const data = await response.json();
    setExistingDatasets(data);
  }

  // -------------------------------------------------------------------------
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

  // -------------------------------------------------------------------------
  function handleClassChange() {
    let classes = [];
    classesTextfields.forEach((textfield) => {
      if (document.getElementById("class-textfield" + textfield).value) {
        classes.push(
          document.getElementById("class-textfield" + textfield).value
        );
      }
    });

    setClasses(classes);
    return classes;
  }

  // -------------------------------------------------------------------------
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
    let localTrainDataPercentage = "";
    if (separateTrainAndTestCheckbox) {
      localTrainDataPercentage =
        document.getElementById("percentage-field").value !== ""
          ? Number(document.getElementById("percentage-field").value)
          : 0.7;

      if (isNaN(localTrainDataPercentage)) {
        setPercentageError(true);
        setPercentageErrorMessage("percentage must be a number (between 0-1)");
        return;
      }

      if (localTrainDataPercentage < 0 || localTrainDataPercentage > 1) {
        setPercentageError(true);
        setPercentageErrorMessage("percentage must be between 0-1");
        return;
      }
    }

    setTrainDataPercentage(localTrainDataPercentage);

    // label column logic
    let localLabelColumn = "";
    if (labeledRadioValue === "yes") {
      localLabelColumn = document.getElementById("label-column-field").value;

      if (localLabelColumn === "") {
        setLabelColumnError(true);
        setLabelColumnErrorMessage(
          "labeled dataset selected. provide a target"
        );
        return;
      }
    }

    setLabelColumn(localLabelColumn);

    // classes logic
    let classes = handleClassChange();

    if (classes.length !== classesTextfields.length) {
      setClassesTextfieldsError(true);
      setClassesTextfieldsErrorMessage("add a value for each class textfield");
      return;
    }

    if (classes.length < 2) {
      setClassesTextfieldsError(true);
      setClassesTextfieldsErrorMessage("provide at least 2 classes");
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
      label: labeledRadioValue === "yes" ? localLabelColumn : -1,
      isSupervised: isSupervisedCheckbox,
      shuffleRows: shuffleRows,
      separateTrainAndTest: separateTrainAndTestCheckbox,
      trainDataPercentage: separateTrainAndTestCheckbox
        ? localTrainDataPercentage
        : -1,
      classes: classes,
      normalClass: normalClass,
      indexOfNormalClass: classes.indexOf(normalClass),
      epochs: epochs,
    };

    setDialogText(JSON.stringify(data, null, "\t"));

    setDialogOpen(true);
    return;
  }

  // -------------------------------------------------------------------------
  async function onFileUploadConfirm() {
    const formData = new FormData();

    for (let i = 0; i < selectedFiles.length; i++) {
      formData.append(`file${i}`, selectedFiles[i]);
    }

    formData.append("is_labeled", labeledRadioValue === "yes");
    formData.append("class_names", classes);
    formData.append("label_column_name", labelColumn);
    formData.append("desired_label", normalClass);
    formData.append(
      "numerical_value_of_desired_label",
      classes.indexOf(normalClass)
    );

    formData.append("separate_train_and_test", separateTrainAndTestCheckbox);
    formData.append("percentage_of_split ", trainDataPercentage);
    formData.append("shuffle_rows", shuffleRows);

    const solutionNature = isSupervisedCheckbox ? "supervised" : "unsupervised";
    formData.append("solution_nature", solutionNature);
    formData.append("dataset_origin", "new_dataset");
    formData.append("model_train_epochs", epochs);

    formData.append("saveData", saveDataCheckbox);

    sendUploadRequest(formData);
  }

  // -------------------------------------------------------------------------
  function handleEKGMethodsCheckboxes(methodNumber) {
    let tempSelectedMethods = selectedMethods;

    if (tempSelectedMethods.includes(methodNumber)) {
      tempSelectedMethods.splice(tempSelectedMethods.indexOf(methodNumber), 1);
    } else tempSelectedMethods.push(methodNumber);

    setSelectedMethods(tempSelectedMethods.sort());
  }

  // -------------------------------------------------------------------------
  async function onUseThisDataset() {
    setDatasetError(false);
    setDatasetErrorMessage("");

    console.log(selectedMethods);

    if (selectedDataset === "") {
      setDatasetError(true);
      setDatasetErrorMessage("you need to select a data-set first...");
      return;
    }

    if (selectedDataset.includes("EKG") && selectedMethods.length === 0) {
      setDatasetError(true);
      setDatasetErrorMessage("you need to select a method for ekg first");
      return;
    }

    if (!selectedDataset.includes("EKG")) handleEKGMethodsCheckboxes(1);

    if (selectedMethods.length === 0) handleEKGMethodsCheckboxes(1);

    let data = {
      dataset: selectedDataset,
      methods: selectedMethods,
      solutionType:
        selectedMethods.length > 1 ? "compareSolutions" : "retrieveData",
    };

    setDialogText(JSON.stringify(data, null, "\t"));

    setDialogOpen(true);
    return;
  }

  // -------------------------------------------------------------------------
  async function onUseThisDatasetConfirm() {
    const formData = new FormData();

    const applicationMode =
      selectedMethods.length > 1 ? "retrieve_data" : "compare_solutions";

    formData.append("application_mode", applicationMode);
    formData.append("dataset_category", selectedDataset);
    formData.append("solution_index", selectedMethods);

    sendUploadRequest(formData);
  }

  // -------------------------------------------------------------------------
  function handleResults(textResponse) {
    const data = JSON.parse(textResponse);

    // setBackendMLPlots(data.plots);
    // setBackendConsole(data.console.split("\n"));
    setBackendResults(data.results);
  }

  // -------------------------------------------------------------------------
  async function loadingResultsScreen() {
    setShowResults(false);
    setLoading(true);

    setTimeout(() => setLoadingText("processing."), 0);
    setTimeout(() => setLoadingText("processing.."), 500);
    setTimeout(() => setLoadingText("processing..."), 1000);

    await delay(1500);

    setLoading(false);
    setShowResults(true);
  }

  // -------------------------------------------------------------------------
  async function sendUploadRequest(formData) {
    await loadingResultsScreen();

    const response = await fetch(BASE_URL + "upload", {
      method: "POST",
      body: formData,
    });

    const textResponse = await response.text();
    setResponseData(textResponse);
    console.log(textResponse);

    // setBackendResults(JSON.parse(textResponse).results);

    handleResults(textResponse);
  }

  // -------------------------------------------------------------------------
  useEffect(() => {
    getExistingDatasetItems();
  }, []);

  return (
    <>
      {showExistingMethod === false &&
      showFileUploadMethod === false &&
      showResults === false ? (
        <>
          <Divv
            style={{
              display: "flex",
              justifyContent: "center",
              marginTop: "35px",
              marginBottom: "5px",
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
              <Tooltip
                title={
                  <Typography fontSize={14}>
                    here you can choose one of the existing dataset that we have
                    provided, such as images or EKGs, and you will get back
                    details about their respective solutions
                  </Typography>
                }
                placement="bottom"
              >
                <Button
                  style={{
                    background:
                      showExistingMethodButtonHover === false
                        ? "black"
                        : "orange",
                    color:
                      showExistingMethodButtonHover === false
                        ? "white"
                        : "black",
                    fontWeight: "bold",
                  }}
                  variant="contained"
                  color="primary"
                  size="large"
                  onMouseEnter={() => setShowExistingMethodButtonHover(true)}
                  onMouseLeave={() => setShowExistingMethodButtonHover(false)}
                  onClick={() => {
                    setShowExistingMethod(true);
                    setShowExistingMethodButtonHover(false);

                    // so no error is shown if there was one previously
                    resetAllFormErrorsAndData();
                  }}
                >
                  use existing dataset
                </Button>
              </Tooltip>
            </Divv>

            <Divv>
              <Tooltip
                title={
                  <Typography fontSize={14}>
                    here you can select one or more files and provide details
                    regarding the classes, the labels, and how you want that
                    data to be divided in order to train a model. afterwards, we
                    shall run it and provide you with some details about how it
                    went
                  </Typography>
                }
                placement="bottom"
              >
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
                  onMouseEnter={() => setShowFileUploadMethodButtonHover(true)}
                  onMouseLeave={() => setShowFileUploadMethodButtonHover(false)}
                  onClick={() => {
                    setShowFileUploadMethod(true);
                    setShowFileUploadMethodButtonHover(false);

                    // so no error is shown if there was one previously
                    resetAllFormErrorsAndData();
                  }}
                >
                  upload a new dataset
                </Button>
              </Tooltip>
            </Divv>
            <Button
              onClick={() => {
                setShowResults(true);
                sendUploadRequest({ id: 1 });
              }}
            >
              Fetch test
            </Button>
          </div>
        </>
      ) : showExistingMethod === true &&
        showFileUploadMethod === false &&
        showResults === false ? (
        <div style={{ textAlign: "center" }}>
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
                position: "absolute",
              }}
              variant="contained"
              color="primary"
              size="large"
              onClick={() => {
                setShowExistingMethod(false);
                setShowFileUploadMethod(false);
                setGoBackButtonHover(false);
              }}
              onMouseEnter={() => setGoBackButtonHover(true)}
              onMouseLeave={() => setGoBackButtonHover(false)}
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
                    console.log("now selected", event.target.value);
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

            {selectedDataset.includes("EKG") && (
              <div style={{ marginBottom: "20px" }}>
                <Tooltip
                  title={
                    <>
                      <Typography
                        fontSize={14}
                        style={{ marginBottom: "5px", padding: "5px" }}
                      >
                        lstm auto-encoder (pytorch)
                      </Typography>
                      <img src={lstmSVG} alt="m1" />
                    </>
                  }
                  placement="bottom"
                  onMouseEnter={() => setMethod1Hover(true)}
                  onMouseLeave={() => setMethod1Hover(false)}
                >
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={selectedMethods.includes(1)}
                        color="default"
                        style={{
                          backgroundColor: method1Hover ? "lightgray" : "",
                          transition: "background 0.4s linear",
                        }}
                        onChange={() => {
                          handleEKGMethodsCheckboxes(1);
                        }}
                      />
                    }
                    label="method #1"
                  />
                </Tooltip>

                <Tooltip
                  title={
                    <>
                      <Typography
                        fontSize={14}
                        style={{ marginBottom: "5px", padding: "5px" }}
                      >
                        convolutional nn (tensorflow/keras)
                      </Typography>
                      <img src={cnnSVG} alt="m2" />
                    </>
                  }
                  placement="bottom"
                  onMouseEnter={() => setMethod2Hover(true)}
                  onMouseLeave={() => setMethod2Hover(false)}
                >
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={selectedMethods.includes(2)}
                        color="default"
                        style={{
                          backgroundColor: method2Hover ? "lightgray" : "",
                          transition: "background 0.4s linear",
                        }}
                        onChange={() => {
                          handleEKGMethodsCheckboxes(2);
                        }}
                      />
                    }
                    label="method #2"
                  />
                </Tooltip>

                <Tooltip
                  title={
                    <Typography fontSize={14}>###PLACEHOLDER3###</Typography>
                  }
                  placement="bottom"
                  onMouseEnter={() => setMethod3Hover(true)}
                  onMouseLeave={() => setMethod3Hover(false)}
                >
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={selectedMethods.includes(3)}
                        color="default"
                        style={{
                          backgroundColor: method3Hover ? "lightgray" : "",
                          transition: "background 0.4s linear",
                        }}
                        onChange={() => {
                          handleEKGMethodsCheckboxes(3);
                        }}
                      />
                    }
                    label="method #3"
                  />
                </Tooltip>
              </div>
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
                onMouseEnter={() => setExistingDatasetButtonHover(true)}
                onMouseLeave={() => setExistingDatasetButtonHover(false)}
              >
                Use this dataset
              </Button>
            </Divv>
          </form>
        </div>
      ) : (
        showResults === false && (
          <div style={{ textAlign: "center" }}>
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
                  position: "absolute",
                }}
                variant="contained"
                color="primary"
                size="large"
                onClick={() => {
                  setShowExistingMethod(false);
                  setShowFileUploadMethod(false);
                  setGoBackButtonHover(false);
                }}
                onMouseEnter={() => setGoBackButtonHover(true)}
                onMouseLeave={() => setGoBackButtonHover(false)}
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
                  width: "40%",
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
                    if (event.target.value !== "yes") {
                      setIsSupervisedCheckbox(false);
                    }
                  }}
                >
                  <FormControlLabel
                    value="yes"
                    control={<Radio />}
                    label="yes"
                  />
                  <FormControlLabel value="no" control={<Radio />} label="no" />
                  <FormControlLabel
                    value="idk"
                    control={<Radio />}
                    label="unsure"
                  />
                  <br></br>
                </RadioGroup>
              </FormControl>

              <Tooltip
                title={
                  labeledRadioValue !== "yes" ? (
                    <Typography fontSize={14}>
                      if the dataset is not labeled, it cannot be supervised
                    </Typography>
                  ) : (
                    ""
                  )
                }
                onMouseEnter={() => setSupervisedCheckboxHover(true)}
                onMouseLeave={() => setSupervisedCheckboxHover(false)}
                placement="bottom"
              >
                <FormControlLabel
                  style={{ margin: "25px", width: "20%" }}
                  control={
                    <Checkbox
                      checked={isSupervisedCheckbox}
                      id="save-data-checkbox"
                      color="default"
                      disabled={labeledRadioValue !== "yes"}
                      style={{
                        backgroundColor:
                          supervisedCheckboxHover && labeledRadioValue !== "yes"
                            ? "lightgray"
                            : "",
                        transition: "background 0.4s linear",
                      }}
                      onChange={() =>
                        setIsSupervisedCheckbox(!isSupervisedCheckbox)
                      }
                    />
                  }
                  label="should it be supervised?"
                />
              </Tooltip>

              <FormControlLabel
                style={{ margin: "25px", width: "20%" }}
                control={
                  <Checkbox
                    checked={shuffleRows}
                    id="save-data-checkbox"
                    color="default"
                    onChange={() => setShuffleRows(!shuffleRows)}
                  />
                }
                label="shuffle rows?"
              />

              <FormControlLabel
                style={{ margin: "25px", width: "20%" }}
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
                          style={{
                            cursor: "pointer",
                            borderRadius: "52px",
                            padding: "15px",
                            margin: "10px",
                            background:
                              removeClassButtonHover === false
                                ? "white"
                                : "orange",
                            transition: "background 0.4s linear",
                          }}
                          onMouseEnter={() => setRemoveClassButtonHover(true)}
                          onMouseLeave={() => setRemoveClassButtonHover(false)}
                          onClick={() => {
                            if (classesTextfields.length === 2) return;
                            setClassesTextfields(
                              classesTextfields.slice(0, -1)
                            );
                            handleClassChange();
                          }}
                        >
                          ((-))
                        </span>
                        what are the classes?
                        <span
                          style={{
                            cursor: "pointer",
                            borderRadius: "52px",
                            padding: "15px",
                            margin: "10px",
                            background:
                              addClassButtonHover === false
                                ? "white"
                                : "orange",
                            transition: "background 0.4s linear",
                          }}
                          onMouseEnter={() => setAddClassButtonHover(true)}
                          onMouseLeave={() => setAddClassButtonHover(false)}
                          onClick={() => {
                            setClassesTextfields([
                              ...classesTextfields,
                              classesTextfields.length + 1,
                            ]);
                            handleClassChange();
                          }}
                        >
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
                    label={"class #" + (textfield - 1)}
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
                    console.log("now selectred", event.target.value);
                    setNormalClass(event.target.value);
                  }}
                >
                  <MenuItem key="0" value="" disabled>
                    choose the normal class
                  </MenuItem>
                  {classes.map((item, index) => {
                    return (
                      <MenuItem
                        key={index}
                        value={
                          item === "" ? "no value at " + (index + 1) : item
                        }
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
                how many epochs to run for?
              </Divv>

              <Slider
                style={{ margin: "25px", width: "40%" }}
                min={5}
                defaultValue={40}
                valueLabelDisplay="on"
                value={epochs}
                onChange={(event, newValue) => setEpochs(newValue)}
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
                onMouseEnter={() => setFileUploadButtonHover(true)}
                onMouseLeave={() => setFileUploadButtonHover(false)}
              >
                Upload file
              </Button>
            </Divv>
          </div>
        )
      )}

      {showResults && (
        <>
          <Divv size="22.5px" style={{ padding: "30px" }}>
            <div
              style={{
                margin: "5px",
                width: "auto",
              }}
            >
              <Terminal name="python outputs">
                {backendConsole.map((line) => {
                  if (line === "") return null;
                  return (
                    <>
                      {">>>"} {line}
                      <br></br>
                    </>
                  );
                })}
              </Terminal>
            </div>

            <Divv>{backendResults}</Divv>

            <Divv left="0px">
              {backendMLPlots.map((src, index) => (
                <Tooltip
                  title={
                    <Typography fontSize={14}>
                      {backendCptions[index]}
                    </Typography>
                  }
                >
                  <img
                    src={src}
                    onClick={() => {
                      setImageViewerOpen(true);
                      setCurrentImage(index);
                    }}
                    width="200"
                    key={index}
                    style={{ margin: "10px", cursor: "pointer" }}
                    alt=""
                  />
                </Tooltip>
              ))}
            </Divv>

            {imageViewerOpen && (
              <ImageViewer
                backgroundStyle={{ backgroundColor: "rgba(0,0,0,0.75)" }}
                src={backendMLPlots}
                currentIndex={currentImage}
                disableScroll={false}
                closeOnClickOutside={true}
                onClose={() => setImageViewerOpen(false)}
              />
            )}
          </Divv>
        </>
      )}

      <div style={{ display: "flex" }}>
        <Dialog open={dialogOpen} maxWidth="xl" fullWidt={true}>
          <DialogTitle style={{ fontWeight: "bold" }}>
            {"proceed with these parameters?"}
          </DialogTitle>
          <DialogContent>
            <DialogContentText style={{ paddingRight: "30px" }}>
              <pre style={{ tabSize: "5" }}>{dialogText}</pre>
            </DialogContentText>
          </DialogContent>
          <DialogActions>
            <Button
              style={{
                background:
                  dialogBackButtonHover === false ? "black" : "orange",
                color: dialogBackButtonHover === false ? "white" : "black",
                fontWeight: "bold",
              }}
              variant="contained"
              color="primary"
              size="large"
              onMouseEnter={() => setDialogBackButtonHover(true)}
              onMouseLeave={() => setDialogBackButtonHover(false)}
              onClick={() => {
                setDialogBackButtonHover(false);
                setDialogOpen(false);
              }}
            >
              Back to data
            </Button>
            <Button
              style={{
                background:
                  dialogConfirmButtonHover === false ? "black" : "orange",
                color: dialogConfirmButtonHover === false ? "white" : "black",
                fontWeight: "bold",
              }}
              variant="contained"
              color="primary"
              size="large"
              onMouseEnter={() => setDialogConfirmButtonHover(true)}
              onMouseLeave={() => setDialogConfirmButtonHover(false)}
              onClick={() => {
                if (
                  showExistingMethod === false &&
                  showFileUploadMethod === true
                )
                  onFileUploadConfirm();
                else onUseThisDatasetConfirm();

                setDialogConfirmButtonHover(false);
                setDialogOpen(false);
              }}
              autoFocus
            >
              Confirm
            </Button>
          </DialogActions>
        </Dialog>
      </div>

      <div>
        <Backdrop
          sx={{ color: "#fff", zIndex: (theme) => theme.zIndex.drawer + 1 }}
          open={loading}
        >
          <CircularProgress color="inherit" />
          <Divv color="white">{loadingText}</Divv>
        </Backdrop>
      </div>

      <WebSocketComponent
        onOutputUpdated={(data) => {
          console.log("io:", data);
          setBackendConsole((backendConsole) => [...backendConsole, data]);
          if (data.toLowerCase().includes("created picture at")) {
            let imageData = data.split("||");

            setBackendMLPlots((backendMLPlots) => [
              ...backendMLPlots,
              imageData[1].trim(),
            ]);

            setBackendCptions((backendCaptions) => [
              ...backendCaptions,
              imageData[2].trim(),
            ]);
          }
        }}
      />
    </>
  );
}
