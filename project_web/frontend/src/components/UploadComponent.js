// TODO stuff
// add all necessary processing for unarchiving and saving the new adress

import { Fragment, useEffect, useState } from "react";

import { Divv, TextFieldFlex, Label } from "./StyledComponents";
import WebSocketComponent from "./WebSocketComponent";

import lstmSVG from "../lstm.svg";
import cnnSVG from "../cnn.svg";

import Terminal from "react-terminal-ui";

import ImageViewer from "react-simple-image-viewer";

import useStore from "./store";

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
  Card,
  CardHeader,
  Collapse,
  CardContent,
  FormGroup,
} from "@mui/material";

const BASE_URL = process.env.REACT_APP_BACKEND;
const nbsps = <>&nbsp;&nbsp;&nbsp;&nbsp;</>;

export default function UploadComponent() {
  // useStore variables
  const terminalFontSize = useStore((state) => state.terminalFontSize);

  // show/hide elements
  const [showExistingMethod, setShowExistingMethod] = useState(false);
  const [showFileUploadMethod, setShowFileUploadMethod] = useState(false);

  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogText, setDialogText] = useState("");

  const [loading, setLoading] = useState(false);
  const [loadingText, setLoadingText] = useState("");

  const [showResults, setShowResults] = useState(false);

  const [showEda, setShowEda] = useState(false);

  const [showEdaButton, setShowEdaButton] = useState(false);

  // data
  const [existingDatasets, setExistingDatasets] = useState([]);

  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedMethods, setSelectedMethods] = useState([]);

  const [method1Selected, setMethod1Selected] = useState(false);
  const [method2Selected, setMethod2Selected] = useState(false);
  const [method3Selected, setMethod3Selected] = useState(false);

  const [methodsError, setMethodsError] = useState(false);

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
  const [beDatasetPath, setBEDatasetPath] = useState("");
  const [eda, setEda] = useState([]);

  const [fileEdaShow, setFileEdaShow] = useState([]);
  const [fileEdaShowHover, setFileEdaShowHover] = useState([]);

  const [edaRequestStarted, setEdaRequestStarted] = useState(false);
  const [edaRequestCompleted, setEdaRequestCompleted] = useState(false);

  const [uploadRequestStarted, setUploadRequestStarted] = useState(false);
  const [uploadRequestCompleted, setUploadRequestCompleted] = useState(false);

  const [edaRequestError, setEdaRequestError] = useState(false);

  const [modelOrigin, setModelOrigin] = useState("use_existing_model");

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

  const [supervisedCheckboxHover, setSupervisedCheckboxHover] = useState(false);

  const [edaConfirmButtonHover, setEdaConfirmButtonHover] = useState(false);
  const [showEdaButtonHover, setShowEdaButtonHover] = useState(false);

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
  const [triggerUseEffect, setTriggerUseEffect] = useState(false);

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
  function getImageIndexFromPath(path) {
    let index = 0;
    for (let i = 0; i < eda.length; i++) {
      for (let j = 0; j < eda[i].plots.length; j++) {
        if (eda[i].plots[j].path === path) {
          return index;
        } else index = index + 1;
      }
    }
    return -1;
  }

  // -------------------------------------------------------------------------
  async function onFileChange(event) {
    if (event.target.files.length === 0) return;
    // console.log([...event.target.files]);

    setBackendConsole([]);
    setBackendMLPlots([]);
    setBackendCptions([]);
    setBackendResults("");
    setEda([]);

    setEdaRequestStarted(false);
    setEdaRequestCompleted(false);

    resetAllFormErrorsAndData();
    setSelectedFiles([...event.target.files]);

    let archiveFound = false,
      multipleFiles = false;

    if (event.target.files.length > 1) multipleFiles = true;

    [...event.target.files].forEach((file) => {
      if (file.name.includes(".zip") || file.name.includes(".rar")) {
        archiveFound = true;
      }
    });

    if (archiveFound && multipleFiles) {
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

    // eda-request-for-new-files
    await loadingResultsScreen("loading eda");
    setShowEda(true);

    const formData = new FormData();

    for (let i = 0; i < [...event.target.files].length; i++) {
      formData.append(`file${i}`, [...event.target.files][i]);
    }

    setEdaRequestStarted(true);

    const response = await fetch(BASE_URL + "perform_eda", {
      method: "POST",
      body: formData,
    });

    const textResponse = await response.text();
    setResponseData(textResponse);

    if (response.status === 400) setEdaRequestError(true);
    else setEdaRequestError(false);

    setEdaRequestCompleted(true);

    const data = JSON.parse(textResponse);

    setBackendResults(data.results);
    setBEDatasetPath(data.path);

    console.log();

    let edaData = [];
    data.eda.forEach((fileInEda) => {
      edaData.push(fileInEda);
    });

    var tempEdaArray = JSON.parse(JSON.stringify(data.eda));

    setFileEdaShow(Array.from({ length: data.eda.length }, () => false));
    setFileEdaShowHover(Array.from({ length: data.eda.length }, () => false));
    // setEda([...data.eda]);
    // setEda(tempEdaArray);
    setEda(data.eda);
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
    setMethodsError(false);

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

    // methods logic
    let localSelectedMethods = [];

    if (method1Selected) localSelectedMethods.push(1);
    if (method2Selected) localSelectedMethods.push(2);
    if (method3Selected) localSelectedMethods.push(3);

    if (localSelectedMethods.length === 0) {
      setMethodsError(true);
      return;
    }

    setSelectedMethods(localSelectedMethods);

    let files = selectedFiles.map((file) => file.name);

    let dialogData = {
      files: files,
      is_labeled: labeledRadioValue === "idk" ? "unknown" : labeledRadioValue,
      label: labeledRadioValue === "yes" ? localLabelColumn : -1,
      is_supervised: isSupervisedCheckbox,
      shuffle_rows: shuffleRows,
      separate_train_and_test: separateTrainAndTestCheckbox,
      train_data_percentage: separateTrainAndTestCheckbox
        ? localTrainDataPercentage
        : -1,
      classes: classes,
      normal_class: normalClass,
      index_of_normal_class: classes.indexOf(normalClass),
      dataset_origin: modelOrigin,
      methods: localSelectedMethods,
      epochs: epochs,
    };

    setDialogText(JSON.stringify(dialogData, null, "\t"));

    setDialogOpen(true);
    return;
  }

  // -------------------------------------------------------------------------
  async function onFileUploadConfirm() {
    // upload-request for new files

    const formData = new FormData();

    // for (let i = 0; i < selectedFiles.length; i++) {
    //   formData.append(`file${i}`, selectedFiles[i]);
    // }

    formData.append("dataset_path", beDatasetPath);

    formData.append("is_labeled", labeledRadioValue === "yes");
    if (labeledRadioValue === "yes")
      formData.append("label_column_name", labelColumn);

    formData.append("class_names", classes);
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
    formData.append("dataset_origin", modelOrigin);
    formData.append("model_train_epochs", epochs);
    formData.append("solution_index", selectedMethods);

    formData.append("save_data", saveDataCheckbox);
    formData.append("clear_images", true);

    sendUploadRequest(formData);
  }

  // -------------------------------------------------------------------------
  function handleEKGMethodsCheckboxes(methodNumber) {
    let tempSelectedMethods = selectedMethods;

    if (tempSelectedMethods.includes(methodNumber)) {
      tempSelectedMethods.splice(tempSelectedMethods.indexOf(methodNumber), 1);
    } else tempSelectedMethods.push(methodNumber);

    // tempSelectedMethods.push(methodNumber);

    return tempSelectedMethods.sort();
  }

  // -------------------------------------------------------------------------
  async function onDatasetSelect(value) {
    setDatasetError(false);
    setShowEdaButton(false);

    setSelectedDataset(value);
    setSelectedMethods([]);

    setSelectedFiles([]);

    setMethod1Selected(false);
    setMethod2Selected(false);
    setMethod3Selected(false);

    setBackendConsole([]);
    setBackendMLPlots([]);
    setBackendCptions([]);
    setBackendResults("");
    setEda([]);

    let localSelectedMethod = value
      .replace(" ", "")
      .replace("#", "")
      .toLowerCase();

    console.log("selected method is", localSelectedMethod);

    // eda-request-for-existing-datasets
    await loadingResultsScreen("loading eda");
    setShowEda(true);

    const formData = new FormData();
    formData.append("dataset_category", localSelectedMethod);
    formData.append("solution_category", localSelectedMethod);

    setEdaRequestStarted(true);

    const response = await fetch(BASE_URL + "perform_eda", {
      method: "POST",
      body: formData,
    });

    const textResponse = await response.text();
    setResponseData(textResponse);

    if (response.status === 400) setEdaRequestError(true);
    else setEdaRequestError(false);

    setEdaRequestCompleted(true);

    const data = JSON.parse(textResponse);

    setBackendResults(data.results);
    setBEDatasetPath(data.path);

    let edaData = [];
    data.eda.forEach((fileInEda) => {
      edaData.push(fileInEda);
    });

    setFileEdaShow(Array.from({ length: data.eda.length }, () => false));
    setFileEdaShowHover(Array.from({ length: data.eda.length }, () => false));
    // setEda([...data.eda]);
    setEda(data.eda);
  }

  // -------------------------------------------------------------------------
  async function onUseThisDataset() {
    setDatasetError(false);
    setDatasetErrorMessage("");

    let localSelectedMethods = [];

    if (selectedDataset === "") {
      setDatasetError(true);
      setDatasetErrorMessage("you need to select a data-set first");
      return;
    }

    if (method1Selected) localSelectedMethods.push(1);
    if (method2Selected) localSelectedMethods.push(2);
    if (method3Selected) localSelectedMethods.push(3);

    if (selectedDataset.includes("EKG") && localSelectedMethods.length === 0) {
      setDatasetError(true);
      setDatasetErrorMessage("you need to select a method for ekg first");
      return;
    }

    if (!selectedDataset.includes("EKG")) localSelectedMethods.push(1);

    if (localSelectedMethods.length === 0) localSelectedMethods.push(1);

    setSelectedMethods(localSelectedMethods);

    let dialogData = {
      dataset: selectedDataset,
      methods: localSelectedMethods,
      solution_type:
        localSelectedMethods.length > 1 ? "compare_solutions" : "retrieve_data",
      model_origin: "use_existing_model",
    };

    setDialogText(JSON.stringify(dialogData, null, "\t"));

    setDialogOpen(true);
    return;
  }

  // -------------------------------------------------------------------------
  async function onUseThisDatasetConfirm() {
    // upload-request for existing datasets
    const formData = new FormData();

    const applicationMode =
      selectedMethods.length > 1 ? "retrieve_data" : "compare_solutions";

    formData.append("application_mode", applicationMode);
    formData.append("dataset_category", selectedDataset);
    formData.append("solution_index", selectedMethods);

    formData.append("clear_images", true);

    sendUploadRequest(formData);
  }

  // -------------------------------------------------------------------------
  async function sendUploadRequest(formData) {
    setBackendConsole([]);
    setBackendMLPlots([]);
    setBackendCptions([]);
    setBackendResults("");

    await loadingResultsScreen("processing");
    setShowResults(true);

    setUploadRequestStarted(true);

    const response = await fetch(BASE_URL + "upload", {
      method: "POST",
      body: formData,
    });

    const textResponse = await response.text();
    setResponseData(textResponse);

    setUploadRequestCompleted(true);

    handleResults(textResponse);
  }

  // -------------------------------------------------------------------------
  async function loadingResultsScreen(loadingText = "processing") {
    setShowResults(false);
    setLoading(true);

    setTimeout(() => setLoadingText(loadingText + "."), 0);
    setTimeout(() => setLoadingText(loadingText + ".."), 500);
    setTimeout(() => setLoadingText(loadingText + "..."), 1000);

    await delay(1500);

    setLoading(false);
  }

  // -------------------------------------------------------------------------
  function handleResults(textResponse) {
    const data = JSON.parse(textResponse);

    // setBackendMLPlots(data.plots);
    // setBackendConsole(data.console.split("\n"));
    setBackendResults(data.results);
  }

  // -------------------------------------------------------------------------
  useEffect(() => {
    getExistingDatasetItems();
  }, []);

  useEffect(() => {}, [triggerUseEffect]);

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
                    here you can choose one of the existing datasets that we
                    have provided, such as images or EKGs running on already
                    existing models, and you will get back details about their
                    respective solutions
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

            {false && (
              <Button
                onClick={() => {
                  setShowResults(true);
                  sendUploadRequest({ id: 1 });
                }}
              >
                Fetch test
              </Button>
            )}
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
                setShowEdaButton(false);
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
                  onChange={(event) => onDatasetSelect(event.target.value)}
                >
                  <MenuItem
                    key="0"
                    value=""
                    style={{ justifyContent: "center" }}
                    disabled
                  >
                    choose a dataset
                  </MenuItem>
                  {existingDatasets.map((item) => {
                    return (
                      <MenuItem
                        key={item.id}
                        value={item.dataset}
                        style={{ justifyContent: "center" }}
                      >
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
                >
                  <FormControlLabel
                    style={{ margin: "25px", width: "10%" }}
                    control={
                      <Checkbox
                        checked={method1Selected}
                        color="default"
                        onChange={() => {
                          setMethod1Selected(!method1Selected);
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
                >
                  <FormControlLabel
                    style={{ margin: "25px", width: "10%" }}
                    control={
                      <Checkbox
                        checked={method2Selected}
                        color="default"
                        onChange={() => {
                          setMethod2Selected(!method2Selected);
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
                >
                  <FormControlLabel
                    style={{ margin: "25px", width: "10%" }}
                    control={
                      <Checkbox
                        checked={method3Selected}
                        color="default"
                        onChange={() => {
                          setMethod3Selected(!method3Selected);
                        }}
                      />
                    }
                    label="method #3"
                  />
                </Tooltip>
              </div>
            )}

            <Divv top="0px">
              {showEdaButton && (
                <Button
                  style={{
                    background:
                      showEdaButtonHover === false ? "black" : "orange",
                    color: showEdaButtonHover === false ? "white" : "black",
                    fontWeight: "bold",
                    marginRight: "15px",
                  }}
                  variant="contained"
                  color="primary"
                  size="large"
                  onClick={() => {
                    setShowEdaButtonHover(false);
                    setShowEda(true);
                  }}
                  onMouseEnter={() => setShowEdaButtonHover(true)}
                  onMouseLeave={() => setShowEdaButtonHover(false)}
                >
                  show eda again
                </Button>
              )}
              <Tooltip
                title={
                  (edaRequestError === true ||
                    (edaRequestStarted === true &&
                      edaRequestCompleted === false)) && (
                    <Typography fontSize={14}>
                      {edaRequestError
                        ? "eda request error. check file type"
                        : "eda is still taking place"}
                    </Typography>
                  )
                }
                placement="bottom"
                arrow={false}
              >
                <Button
                  style={{
                    background:
                      edaRequestError ||
                      (edaRequestStarted === true &&
                        edaRequestCompleted === false)
                        ? "gray"
                        : existingDatasetButtonHover === false
                        ? "black"
                        : "orange",
                    color:
                      existingDatasetButtonHover === false ? "white" : "black",
                    fontWeight: "bold",
                  }}
                  disabled={
                    edaRequestError === true ||
                    loading === true ||
                    (edaRequestStarted === true &&
                      edaRequestCompleted === false)
                  }
                  variant="contained"
                  color="primary"
                  size="large"
                  onClick={() => onUseThisDataset()}
                  onMouseEnter={() => setExistingDatasetButtonHover(true)}
                  onMouseLeave={() => setExistingDatasetButtonHover(false)}
                >
                  use this dataset
                </Button>
              </Tooltip>
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
                : <br></br>
                <Tooltip
                  title={
                    <Typography fontSize={14}>
                      click to check eda again
                    </Typography>
                  }
                >
                  <span
                    style={{
                      textDecorationLine: "underline",
                      cursor: "pointer",
                    }}
                    onClick={() => {
                      setShowEda(true);
                    }}
                  >
                    {stringOfFilesUploaded}
                  </span>
                </Tooltip>
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
                  width: "25%",
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
                  labeledRadioValue !== "yes" && (
                    <Typography fontSize={14}>
                      if the dataset is not labeled, it cannot be supervised
                    </Typography>
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
                  separateTrainAndTestCheckbox && (
                    <Typography fontSize={14}>default value is 0.7</Typography>
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
                    {textfield === 1 && (
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
                    // console.log("now selected", event.target.value);
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
              >
                <FormControlLabel
                  style={{
                    margin: "25px",
                    width: "10%",
                    color: methodsError ? "red" : "",
                  }}
                  control={
                    <Checkbox
                      style={{ color: methodsError ? "red" : "" }}
                      checked={method1Selected}
                      color="default"
                      onChange={() => {
                        setMethod1Selected(!method1Selected);
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
              >
                <FormControlLabel
                  style={{
                    margin: "25px",
                    width: "10%",
                    color: methodsError ? "red" : "",
                  }}
                  control={
                    <Checkbox
                      style={{ color: methodsError ? "red" : "" }}
                      checked={method2Selected}
                      color="default"
                      onChange={() => {
                        setMethod2Selected(!method2Selected);
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
              >
                <FormControlLabel
                  style={{
                    margin: "25px",
                    width: "10%",
                    color: methodsError ? "red" : "",
                  }}
                  control={
                    <Checkbox
                      style={{ color: methodsError ? "red" : "" }}
                      checked={method3Selected}
                      color="default"
                      onChange={() => {
                        setMethod3Selected(!method3Selected);
                      }}
                    />
                  }
                  label="method #3"
                />
              </Tooltip>
            </div>

            <FormControl
              style={{
                margin: "25px",
                marginTop: "0px",
                display: "flex",
                alignItems: "center",
              }}
            >
              <RadioGroup
                row
                defaultValue="use_existing_model"
                value={modelOrigin}
                onChange={(event) => setModelOrigin(event.target.value)}
              >
                <FormControlLabel
                  value="use_existing_model"
                  control={<Radio />}
                  label="use an existing model if found"
                />
                <Tooltip
                  placement="right"
                  title={<Typography fontSize={14}>please don't</Typography>}
                >
                  <FormControlLabel
                    value="train_new_model"
                    control={<Radio />}
                    label="train new model"
                  />
                </Tooltip>
              </RadioGroup>
            </FormControl>

            <Tooltip
              title={
                (edaRequestError === true ||
                  (edaRequestStarted === true &&
                    edaRequestCompleted === false)) && (
                  <Typography fontSize={14}>
                    {edaRequestError
                      ? "eda request error. check file type"
                      : "eda is still taking place"}
                  </Typography>
                )
              }
              placement="bottom"
              arrow={false}
            >
              <Divv>
                <Button
                  style={{
                    background:
                      edaRequestError ||
                      (edaRequestStarted === true &&
                        edaRequestCompleted === false)
                        ? "gray"
                        : fileUploadButtonHover === false
                        ? "black"
                        : "orange",
                    color: fileUploadButtonHover === false ? "white" : "black",
                    fontWeight: "bold",
                  }}
                  disabled={
                    edaRequestError === true ||
                    loading === true ||
                    (edaRequestStarted === true &&
                      edaRequestCompleted === false)
                  }
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
            </Tooltip>
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
              <Terminal id="results-terminal" name="python outputs">
                {backendConsole.map((line) => {
                  if (line === "") return null;
                  return (
                    <span style={{ fontSize: terminalFontSize }}>
                      {">>>"} {line}
                      <br></br>
                    </span>
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
        <Dialog
          open={dialogOpen}
          maxWidth="xl"
          fullWidth={true}
          scroll={"body"}
        >
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

      {showEda && (
        <Dialog open={showEda} maxWidth="xl" fullWidth={true}>
          <DialogContent>
            <Terminal
              id="eda-terminal"
              name="python outputs"
              style={{
                margin: "5px",
                width: "auto",
              }}
            >
              {backendConsole.map((line) => {
                if (line === "") return null;

                return (
                  <span style={{ fontSize: terminalFontSize }}>
                    {">>>"} {line}
                    <br></br>
                  </span>
                );
              })}
            </Terminal>

            <Divv>{backendResults}</Divv>

            {eda.map((fileData, index) => {
              console.log("eda-rerendered");
              return (
                <Card
                  style={{
                    margin: "20px",
                    backgroundColor: fileEdaShow[index] ? "#eeeee4" : "#eeeeee",

                    cursor: "pointer",
                  }}
                  onMouseEnter={() => {
                    setFileEdaShowHover((oldFileEdaShow) => {
                      const newFileEdaShow = Array.from(
                        { length: oldFileEdaShow.length },
                        () => false
                      );
                      newFileEdaShow[index] = true;
                      return newFileEdaShow;
                    });
                  }}
                  onMouseLeave={() => {
                    setFileEdaShowHover((oldFileEdaShow) => {
                      const newFileEdaShow = Array.from(
                        { length: oldFileEdaShow.length },
                        () => false
                      );
                      newFileEdaShow[index] = false;
                      return newFileEdaShow;
                    });
                  }}
                >
                  <CardHeader
                    title={
                      "file #" + fileData.index + " --- " + fileData.filename
                    }
                    onClick={() => {
                      setFileEdaShow((oldFileEdaShow) => {
                        const newFileEdaShow = Array.from(
                          { length: oldFileEdaShow.length },
                          () => false
                        );
                        newFileEdaShow[index] = !oldFileEdaShow[index];
                        return newFileEdaShow;
                      });
                    }}
                  />
                  <Collapse in={fileEdaShow[index]}>
                    <CardContent>
                      <Typography paragrah>
                        <span style={{ fontWeight: "bold" }}>shape</span> ---{" "}
                        {fileData.rows} rows, {fileData.columns} columns
                      </Typography>
                      <br></br>
                      <Typography paragrah>
                        <span style={{ fontWeight: "bold" }}>
                          columns with missing data
                        </span>{" "}
                        --- {fileData.columns_with_missing_data} column(s)
                      </Typography>
                      <br></br>
                      {/* 
                      <Typography paragrah>
                        <span style={{ fontWeight: "bold" }}>info</span> ---{" "}
                        {fileData.info}
                      </Typography>
                      <br></br>
                       */}
                      <Typography paragrah>
                        <span style={{ fontWeight: "bold" }}>head of file</span>{" "}
                        ---{" "}
                        <table>
                          <tbody>
                            {fileData.head.map((row, rowIndex) => (
                              <tr key={rowIndex}>
                                {row.split(/\s+/).map((item, itemIndex) => (
                                  <td
                                    style={{ paddingRight: "15px" }}
                                    key={itemIndex}
                                  >
                                    {item}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </Typography>

                      <br></br>
                      <Typography paragrah>
                        <span style={{ fontWeight: "bold" }}>
                          description of data
                        </span>{" "}
                        ---
                        <table>
                          <tbody>
                            {fileData.describe.map((row, rowIndex) => (
                              <tr key={rowIndex}>
                                {row.split(/\s+/).map((item, itemIndex) => (
                                  <td
                                    style={{ paddingRight: "15px" }}
                                    key={itemIndex}
                                  >
                                    {item}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </Typography>

                      <Divv left="0px">
                        {fileData.plots.map((plot, iindex) => {
                          return (
                            <Tooltip
                              title={
                                <Typography fontSize={14}>
                                  {plot.caption}
                                </Typography>
                              }
                            >
                              <img
                                src={plot.path}
                                onClick={() => {
                                  setImageViewerOpen(true);
                                  setCurrentImage(
                                    getImageIndexFromPath(plot.path)
                                  );
                                }}
                                width="150"
                                key={index}
                                style={{
                                  margin: "10px",
                                  cursor: "pointer",
                                }}
                                alt=""
                              />
                            </Tooltip>
                          );
                        })}
                      </Divv>
                    </CardContent>
                  </Collapse>
                </Card>
              );
            })}

            {imageViewerOpen && (
              <ImageViewer
                backgroundStyle={{
                  backgroundColor: "rgba(0,0,0,0.75)",
                }}
                src={eda.flatMap((item) => item.plots).map((plot) => plot.path)}
                currentIndex={currentImage}
                disableScroll={false}
                closeOnClickOutside={true}
                onClose={() => setImageViewerOpen(false)}
              />
            )}
          </DialogContent>
          <DialogActions>
            <Button
              style={{
                background:
                  edaConfirmButtonHover === false ? "black" : "orange",
                color: edaConfirmButtonHover === false ? "white" : "black",
                fontWeight: "bold",
              }}
              variant="contained"
              color="primary"
              size="large"
              onMouseEnter={() => setEdaConfirmButtonHover(true)}
              onMouseLeave={() => setEdaConfirmButtonHover(false)}
              onClick={() => {
                setEdaConfirmButtonHover(false);
                setShowEdaButton(true);
                setShowEda(false);
              }}
            >
              got it
            </Button>
          </DialogActions>
        </Dialog>
      )}

      <WebSocketComponent
        onOutputUpdated={(data) => {
          // console.log("io:", data);

          if (data.includes("\n") === false)
            setBackendConsole((backendConsole) => [...backendConsole, data]);
          else {
            const splitLines = data.split("\n");
            splitLines.forEach((splitLine) => {
              setBackendConsole((backendConsole) => [
                ...backendConsole,
                splitLine,
              ]);
            });
          }

          if (data.toLowerCase().includes("created picture")) {
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
